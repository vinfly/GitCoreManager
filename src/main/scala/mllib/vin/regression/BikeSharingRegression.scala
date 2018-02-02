package mllib.vin.regression

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD, RegressionModel}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel


/**
  * 使用Spark构建回归模型
  */
object BikeSharingRegression {


  /**
    *
    * @param args      传递参数,从Main方法传递过来
    * @param operation 贷出模式中的用户函数,真正处理数据的地方
    */
  def sparkOperation(args: Array[String])(operation: SparkSession => Unit): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .appName("BikeSharingRegression")
      .master("local[4]")
      .getOrCreate()
    //设置日志级别信息
    spark.sparkContext.setLogLevel("WARN")
    try {
      //调用用户函数处理数据
      operation(spark)
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      spark.stop()
    }
  }

  def modelTrain(spark: SparkSession): Unit = {
    //获取SparkContext对象实例
    val sc = spark.sparkContext

    //TODO 获取数据源
    val rawDatasDF: DataFrame = spark.read
      .option("header", "true")
      .csv("/usr/spark/bikesharing/hour.csv")

    //样本数据集的Schema信息
    def printDataBasicInfo(df: DataFrame): Unit = {
      println(s"总的条目数: ${df.count()}")
      //获取前五条数据
      df.show(5)
    }

    printDataBasicInfo(rawDatasDF)

    //随机采样数据,使用sample方法
    //val sampleDS:Dataset[Row] = rawDatasDF.sample(withReplacement = true,0.01,123)
    //sampleDS.collect().foreach(println)
    //    +-------+----------+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+----------+---+
    //    |instant|    dteday|season| yr|mnth| hr|holiday|weekday|workingday|weathersit|temp| atemp| hum|windspeed|casual|registered|cnt|
    //    +-------+----------+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+----------+---+
    //    |      1|2011-01-01|     1|  0|   1|  0|      0|      6|         0|         1|0.24|0.2879|0.81|        0|     3|        13| 16|
    //    |      2|2011-01-01|     1|  0|   1|  1|      0|      6|         0|         1|0.22|0.2727| 0.8|        0|     8|        32| 40|
    /**
      * 经过数据分析:
      *  1. 共17个字段
      *  2. 忽略四个无关特征的字段信息 instant,dteday,casual,registered
      *  3. 提取特征字段有12个
      *    - 8个字段:属于类别类型数据 :将类别数据转换为数值类型的数据
      *    - 4个字段: 属于连续性数值类型数据
      *   4. 一个字段Label预测目标值 cnt
      */
    //将DataFrame转换为RDD进行操作
    val recordsRDD: RDD[Row] = rawDatasDF.rdd
    recordsRDD.persist(StorageLevel.MEMORY_ONLY)

    /**
      * 定义一个函数指定下表,返回Map
      *
      * @param rdd
      * @param index
      * @return
      */
    def getMapping(rdd: RDD[Row], index: Int): collection.Map[String, Long] = {
      rdd.map(row => row.getString(index))
        .distinct()
        //按照自然升序排序
        .sortBy(x => x)
        //拉链操作,给每个类别指定一个下标值
        .zipWithIndex()
        //转换为Map类型数据
        .collectAsMap()
    }

    //测试
    //    val seasonMapping = getMapping(recordsRDD, 2)
    //    println(seasonMapping)
    //Map(2 -> 1, 1 -> 0, 4 -> 3, 3 -> 2)

    /**
      * 将第二列至第九列类别数据转换为二元组对儿
      */
    val mappings: IndexedSeq[collection.Map[String, Long]] = (2 to 9).map {
      index => getMapping(recordsRDD, index)
    }

    //通过广播变量方式将数据广播出去
    val mappingBroadcast: Broadcast[IndexedSeq[collection.Map[String, Long]]] = sc.broadcast(mappings)

    //计算完每个变量映射以后,统计最终二元向量的总长度:用户构造特征
    val categoryLength: Int = mappings.map(_.size).sum
    //可以将类别长度广播出去
    val categoryLengthBroadcast: Broadcast[Int] = sc.broadcast(categoryLength)


    def extractFeatures(record: Seq[String], categoryLen: Int): linalg.Vector = {
      //定义一个数组
      val categoryFeatures = Array.ofDim[Double](categoryLen)

      //针对具体某行记录某一列的类别特征值,填充数据
      var step = 0
      for ((field, index) <- record.slice(2, 10).zipWithIndex) {
        //获取对应列的特征数据的1-of-k 映射
        val mapping: collection.Map[String, Long] = mappingBroadcast.value(index)

        //获取本列的对应位置
        val idx = mapping(field)
        //数组中位置
        val arrIndex = step + idx.toInt
        //设置在数组中的具体位置赋值
        categoryFeatures(arrIndex) = 1.0
        //更新步长
        step += mapping.size
      }

      //提取其它4个正则化的数据:温度,体感温度,湿度,风向
      val otherFeatures: Seq[Double] = record.slice(10, 14).map(_.toDouble)

      //创建向量,稠密向量 大多数的特征都是非零值
      Vectors.dense(categoryFeatures ++ otherFeatures)
    }


    def extractLabel(record: Seq[String]): Double = {
      record.last.toDouble
    }

    /**
      * TODO : 构造出特征向量
      */
    val lpsRDD: RDD[LabeledPoint] = recordsRDD.map {
      row =>
        //将Row类型的数据转换为序列Seq
        val record: Seq[String] = row.toSeq.map(_.toString)
        //提取特征数据
        val features: linalg.Vector = extractFeatures(record, categoryLengthBroadcast.value)
        //提取标签
        val label: Double = extractLabel(record)
        //标签向量
        LabeledPoint(label, features)
    }


    /**
      * 交叉验证测试
      */
    val Array(trainRDD, testRDD) = lpsRDD.randomSplit(Array(0.8, 0.2), seed = 123)


    /**
      * TODO : 选择合适的算法模型,训练模型进行模型训练
      */
    //为了测试方便,封装在函数中
    def trainingModel(trainingRDD: RDD[LabeledPoint]): LinearRegressionModel = {

      //使用训练集训练模型
      val lrModel: LinearRegressionModel = LinearRegressionWithSGD.train(trainingRDD, 100, 0.01)
      lrModel
    }

    //模型预测
    def predictModel(model: RegressionModel, testingRDD: RDD[LabeledPoint]): RDD[(Double, Double)] = {

      //使用测试集数据进行预测结果
      testingRDD.map(lp => {

        //根据每小时数据特征,使用模型预测出租自行车数量
        val predictLabel = model.predict(lp.features)

        //返回实际值与预测值
        (lp.label, predictLabel)
      })

    }

    //模型预测结果
    val acturalAndPredictRDD: RDD[(Double, Double)] = predictModel(trainingModel(trainRDD), testRDD)

    acturalAndPredictRDD.take(10).foreach(println)


    /**
      * TODO 评估回归模型的性能
      *   - 均方误差 MSE:Mean Squared Error
      * 所有样本预测值与实际值差的平方之和除以样本总数
      *   - 均方根误差 RMSE:Root Mean Sqared Error
      * 就是均方误差的平方根
      *   - 平均绝对误差 MAE:Mean Absoluate Error
      * 预测值和实际值的差的绝对值的平均值
      *   - 均方根对数误差
      * 与均方根误差相比,需要对实际值和预测值进行归一化以后,取对数再进行计算
      * log(pred+1)-log(actural+1)
      */

    //模型评估:依据测试数据集预测的结果进行性能评估
    def modelEvaluate(apRDD: RDD[(Double, Double)]): Unit = {
      //由于指标都是均值,所以要获取总数
      val predCount = apRDD.count().toDouble

      //计算MSE:
      val mseValue = apRDD.map {
        case (actural, predict) => Math.pow(actural - predict, 2)
      }.sum() / predCount

      //计算RMSE
      val rmseValue = Math.sqrt(mseValue)

      //计算MAE
      val maeValue = apRDD.map {
        case (actural, predict) => Math.abs(actural - predict)
      }.sum() / predCount

      println(s"MSE: $mseValue , RMSE:$rmseValue , MAE: $maeValue")

      val rmlseValue = Math.sqrt(
        apRDD.map {
          case (actural, predict) => Math.pow(Math.log(actural + 1) - Math.log(predict + 1), 2)
        }.sum() / predCount
      )
      println(s"RMLSE: $rmlseValue")
      println("--------------------------分割线--------------------------------")

    }
    //TODO 模型评估
    modelEvaluate(acturalAndPredictRDD)

    /**
      * TODO : 使用回归模型中的决策树回归算法训练模型
      */

    def extractFeaturesDT(record: Seq[String]): Vector = {

      //提取特征,转换为数组
      val features = record.slice(2, 14).map(_.toDouble).toArray
      //构建稠密向量
      Vectors.dense(features)
    }

    val Array(testDtRDD, trainDtRDD) = recordsRDD.map(row => {
      //将Row类型的数据转换为训练Seq
      val record: Seq[String] = row.toSeq.map(_.toString)
      //提取特征数据
      val features: Vector = extractFeaturesDT(record)
      //提取标签数据
      val label = extractLabel(record)
      //返回标签向量
      LabeledPoint(label, features)

    }).randomSplit(Array(0.2, 0.8), seed = 123L)

    //基于决策树训练模型和评估模型
    def dtTrainAndEvaluateModel(trainDataset: RDD[LabeledPoint],
                                testDataset: RDD[LabeledPoint]): Unit = {
      //使用据册书回归算法训练模型

      val dtModel: DecisionTreeModel = DecisionTree.trainRegressor(
        trainDataset, //训练数据集
        Map[Int, Int](),
        impurity = "variance",
        maxDepth = 10, //5->8->10->12
        maxBins = 8 //32->16
      )

      //使用测试数据集测试模型
      val acturalAndPredictDtRDD = testDataset.map {
        case LabeledPoint(label, features) => (label, dtModel.predict(features))
      }

      //评估决策树模型性能
      modelEvaluate(acturalAndPredictDtRDD)

    }

    //使用类别数据未转化的数据进行测试
    dtTrainAndEvaluateModel(trainDtRDD, testDtRDD)
    //使用转化后的数据
    dtTrainAndEvaluateModel(trainRDD, testRDD)


    /**
      * TODO : 决策树模型性能优化
      * 决策树提供两个主要的参数: 最大的树的深度和最大的划分数
      *  - 1.树深度
      * 树的深度越深,训练越浮渣,也越耗时,不能太大
      *  - 2.最大划分数
      * 可以调整划分数小一点
      */

    /**
      * TODO : 将训练得到的模型,需要保存起来,以便后续进行使用,即模型的持久化
      * - 1.保存持久化
      * 通常持久化到HDFS文件系统
      * -2.在线模型的使用
      * 实际生产环境,模型的测试使用
      */


    def saveAndLoadPredict(model: LinearRegressionModel, sparkContext: SparkContext): Unit = {

      //第一步: 保存模型
      model.save(sparkContext, "datas/regression/biksharing/model/")
      //第二步,加载模型
      val loadLRModel: LinearRegressionModel = LinearRegressionModel
        .load(sparkContext, "datas/regression/bikesharing/model/")
      //第三步,提取特征进行预测使用
      //模拟业务实际数据
      val rawRecord = "33,2011-01-02,1,0,1,9,0,0,0,2,0.38,0.3939,0.76,0.2239,1,19,20"
      val predictRecord: Array[String] = rawRecord.split(",")
      //提取特征
      val features: Vector = extractFeatures(predictRecord, categoryLength)
      //获取Label标签
      val label: Double = extractLabel(predictRecord)
      //使用模型预测,此模型从持久化中加载出来的
      val loadPredictLabel: Double = loadLRModel.predict(features)
      //使用直接训练得到模型,预测
      val predictLabel = model.predict(features)

      println(s"Actural: $label, Predict: $predictLabel, Load Predict: $loadPredictLabel")
    }
    //saveAndLoadPredict(trainingModel(trainRDD),sc)

  }


  def main(args: Array[String]): Unit = {
    sparkOperation(args)(modelTrain)
  }

}
