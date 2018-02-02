package mllib.vin.classification


import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Spark MLlib分类
  *
  * 业务: 用于判断推荐的页面是短暂的(0),还是长久的(1)
  * 设计模式: 使用scala贷出模式进行编程
  * 该模式主要针对资源的开启和关闭而言的一种模式,有两类函数:
  *     1.贷出函数: 管理资源的开启,关闭等
  *     2.用户函数: 真正业务逻辑实现的地方
  */
object MLClassificationTest {

  def sparkOperation(args: Array[String])(operation: SparkContext => Unit): Unit = {
    //    if (args.length != 2) {
    //      println("Usage:<AppName> <Master>")
    //      throw new IllegalArgumentException("Need Two Args ............")
    //    }

    //构建SparkConf实例,设置应用的配置信息
    val sparkConf = new SparkConf()
      .setAppName("MLClassification Application")
      .setMaster("local[4]")
    //创建 SparkContext实例
    val sc = SparkContext.getOrCreate(sparkConf)
    //设置日志级别信息
    sc.setLogLevel("WARN")

    try {
      //调用用户函数处理数据
      operation(sc)
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      sc.stop()
    }
  }


  /**
    * 用户函数
    *
    * @param sc
    */
  def modelTrain(sc: SparkContext): Unit = {

    //TODO 1 : 加载数据
    val rawsRDD: RDD[String] = sc.textFile("/usr/spark/stumbleupon/train_noheader.tsv")
    //采样数据
    // println(s"Count = ${rawsRDD.count()}")
    // println(s"Top 10 \n ${rawsRDD.take(10).mkString("\n")}")

    //按照制表符进行分割数据
    val recordsRDD: RDD[Array[String]] = rawsRDD
      //过滤不符合的数据,数据长度不符合
      .filter(line => {
      line.trim.length > 0 && line.trim.split("\t").length == 27
    })
      .map(_.split("\t").map(_.replaceAll("\"", "")))

    //TODO 2 : 数据清洗和转换
    val labeledPointsRDD: RDD[LabeledPoint] = recordsRDD
      .map(fields => {
        //获取标签 label
        val label = fields.last.toDouble

        //获取每条数据特征features
        val featureArray: Array[Double] = fields
          //从第五个开始获取数据,一直到倒数第二个,def slice(from:Int,until:Int)左闭右开
          .slice(4, fields.length - 1)
          //对缺少值进行转换,此处使用0值进行填充,也可使用其他方式
          .map(fields => if ("?" == fields) 0.0 else fields.toDouble)

        //得到特征向量
        val features: Vector = Vectors.dense(featureArray)

        //得到标签向量(标量) = 向量(Vector) + 标签(label)
        LabeledPoint(label, features)
      })

    //TODO : 由于后续使用数据进行模型训练,迭代多次进行,此处将数据RDD进行缓存
    labeledPointsRDD.persist(StorageLevel.MEMORY_ONLY)


    /**
      * 交叉验证法(训练+测试), 将数据即拆分为两部分,比例8:2
      */

    //调用RDD的randomSplit方法,将数据集随机按权重划分,seed参数可忽略不计
    val datasRDD: Array[RDD[LabeledPoint]] = labeledPointsRDD
      .randomSplit(Array(0.8, 0.2), seed = 123)

    //得到训练数据集
    val trainingDatasRDD: RDD[LabeledPoint] = datasRDD(0)
    trainingDatasRDD.cache()

    //得到测试数据集
    val testDatasRDD: RDD[LabeledPoint] = datasRDD(1)


    //TODO 3 : 调用MLlib库中API进行训练数据

    //使用逻辑回归(降维)
    val lrModel: LogisticRegressionModel = LogisticRegressionWithSGD.train(trainingDatasRDD, 100)

    //使用 支持向量机 模型
   //  val svmModel: SVMModel = SVMWithSGD.train(trainingDatasRDD,100)

    //使用测试集数据测试模型
    val predictVsTrueRDD: RDD[(Double, Double)] = predictModel(lrModel, testDatasRDD)


    //TODO 4 : 模型性能评估
    predictModelAndEvaluate(predictVsTrueRDD)


    //TODO 5 : 改进模型性能及参数调优
    /**
      * 特征标准化(正则化)
      * 增加其它特征
      */

    //研究一下特征数据分布
    //featureStatistics(labeledPointsRDD)
    //对特征数据标准化
    //featureStandard(labeledPointsRDD)

    println("-----------------------------特征数据标准化---------------------------")
    standardModelAndEvaluate(labeledPointsRDD)


  }


  /**
    * 依据所得到的模型,进行预测值
    *
    * @param model        输入建立好的模型
    * @param testDatasRDD 输入的测试数据集
    */
  def predictModel(model: ClassificationModel, testDatasRDD: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    val predictVsTrueRDD: RDD[(Double, Double)] = testDatasRDD.map {
      case LabeledPoint(label, features) =>

        //使用模型预测值
        val predictLabel = model.predict(features)
        //将预测值和真实值返回
        (predictLabel, label)
    }
    predictVsTrueRDD

  }

  /**
    * 根据测试数据的测试结果进行模型的性能评估
    *
    * @param predictsVsTrueRDD
    */
  def predictModelAndEvaluate(predictsVsTrueRDD: RDD[(Double, Double)]): Unit = {

    /**
      * 预测正确率
      */
    val lrTotalCorrect: Double = predictsVsTrueRDD
      .map(item => if (item._1 == item._2) 1 else 0).sum()

    val lrArruracy = lrTotalCorrect / predictsVsTrueRDD.count().toDouble
    println(s"逻辑回归模型预测的正确率: $lrArruracy")


    /**
      * 预测准确率和召回率 PR曲线面积
      */
    // def this(scoreAndLabels: RDD[(Double, Double)]) = this(scoreAndLabels, 0)
    val metrics = new BinaryClassificationMetrics(predictsVsTrueRDD)

    //PR 曲线面积,越接近于1的时候,模型越好 , P:Precision 准确率 , R:Recall 召回率
    val prArea = metrics.areaUnderPR()
    //ROC 曲线面积,越接近于1的时候,模型越好
    val rocArea = metrics.areaUnderROC()

    println(s"PR 曲线面积: $prArea , ROC 曲线面积: $rocArea")

  }

  /**
    * 对数据集进行研究,特征分布状况
    *
    * @param labeledPointsRDD
    */
  def featureStatistics(labeledPointsRDD: RDD[LabeledPoint]): Unit = {

    //获取特征值
    val vectors: RDD[Vector] = labeledPointsRDD.map(lp => lp.features)

    //构建一个矩阵,将RDD转化为一个矩阵
    val matrix = new RowMatrix(vectors)

    //对每一列数据进行概要统计
    val statistics: MultivariateStatisticalSummary = matrix.computeColumnSummaryStatistics()

    //打印矩阵每列均值
    println(s"每列的均值 : ${statistics.mean}")
    //打印每列的方差
    println(s"每列的方差 : ${statistics.variance}")
    //打印每列的最大和最小值
    println(s"每列的最大值 : ${statistics.max} , 每列的最小值 : ${statistics.min}")
    //打印每列的非零项数
    println(s"每列中非0项的数: ${statistics.numNonzeros}")


  }

  /**
    * 对数据集中的每个特征向量进行标准化:
    *   1. 与均值向量按项进行依次减法
    * 每一列的平均值与向量中的特征值(列值)相减
    *    2. 依次按项除以特征的标准差
    * 标准差向量:方差向量的每项平方根得到
    *
    * @param labeledPointsRDD
    */
  def featureStandard(labeledPointsRDD: RDD[LabeledPoint]): Unit = {

    //从标签向量中获取特征值,以向量的形式返回
    val vectors: RDD[Vector] = labeledPointsRDD.map(lp => lp.features)

    //创建 StandardScaler对象,第一个参数:表示是否从数据中减去均值;第二个参数: 表示是否应用标准差进行缩放
    val standardScaler = new StandardScaler(withMean = true, withStd = true)
    //使用StandardScaler对象计算得到特征值的均值和标准差的模型
    val scalerModel: feature.StandardScalerModel = standardScaler.fit(vectors)

    //进行特征值标准化
    val scaledLabeledPointRDD: RDD[LabeledPoint] = labeledPointsRDD
      .map(lp => LabeledPoint(lp.label, scalerModel.transform(lp.features)))

    /**
      * 观察一下标准化前和标准化后的向量
      */
    println(labeledPointsRDD.first().features)
    println(scaledLabeledPointRDD.first().features)


  }

  /**
    * 对特征数据进行标准化,并进行模型训练及模型评估
    *
    * @param labeledPointsRDD
    */
  def standardModelAndEvaluate(labeledPointsRDD: RDD[LabeledPoint]): Unit = {

    /**
      * 1.特征向量标准化
      */
    //从标签向量中获取特征值,以向量的形式返回
    val vectors: RDD[Vector] = labeledPointsRDD.map(lp => lp.features)
    //创建StandardScaler对象,第一个参数:表示是否从数据中减去均值;第二个参数,表示是否应用标准差进行缩放
    val standardScaler = new StandardScaler(withMean = true, withStd = true)
    //使用StandardScaler对象计算得到特征值的均值和标准差的模型
    val scalerModel: feature.StandardScalerModel = standardScaler.fit(vectors)
    //进行特征值标准化
    val scaledlabelPointRDD: RDD[LabeledPoint] = labeledPointsRDD
      .map(lp => LabeledPoint(lp.label, scalerModel.transform(lp.features)))


    //调用RDD的randomSplit方法按比例切分RDD数据
    val datasRDD = scaledlabelPointRDD.randomSplit(Array(0.8, 0.2), seed = 123)
    //得到训练集数据
    val trainingDatasRDD: RDD[LabeledPoint] = datasRDD(0)
    //得到测试集数据
    val testDatasRDD: RDD[LabeledPoint] = datasRDD(1)

    trainingDatasRDD.cache()


    // 使用 逻辑回归（降维）
    val lrModel: LogisticRegressionModel = LogisticRegressionWithSGD.train(trainingDatasRDD, 100)

    val predictVsTrueRDD: RDD[(Double, Double)] = predictModel(lrModel, testDatasRDD)
    //模型性能评估
    predictModelAndEvaluate(predictVsTrueRDD)


    /**
      * TODO
      * 进行模型训练的时候,需要调整参数进行测试
      * 比如迭代次数,步数等
      */
    println("----------------------------调整模型参数进行优化 : 迭代次数-----------------------")
    //TODO : 针对迭代的次数进行测试
    val iterResults: Seq[(String, Double, Double)] = Seq(10, 50, 100, 200, 500, 800, 1000).map {
      param =>
        //模型
        val model: LogisticRegressionModel = trainWithParams(trainingDatasRDD, 0.0, param, 1.0)
        //评估性能
        createMetrics(s"$param iterations", testDatasRDD, model)
    }
    iterResults.foreach {
      case (param, pr, auc) =>
        println(f"$param, PR = ${pr * 100}%2.2f%%, AUC = ${auc * 100}%2.2f%%")
    }


    println("----------------------------调整模型参数进行优化 : 步长-----------------------")
    //TODO : 针对步长进行测试
    val stepResults: Seq[(String, Double, Double)] = Seq(0.001, 0.01, 0.1, 1.0, 10.0, 20.0).map { param =>
      //模型
      val model: LogisticRegressionModel = trainWithParams(trainingDatasRDD, 0.0, 100, param)
      //评估性能
      createMetrics(s"$param step size", testDatasRDD, model)
    }
    stepResults.foreach {
      case (param, pr, auc) =>
        println(f"$param, PR = ${pr * 100}%2.2f%%, AUC = ${auc * 100}%2.2f%%")
    }

  }

  /**
    * 辅助函数,在给定参数之后,训练逻辑回归模型
    *
    * @param value
    * @param regParam
    * @param numIterations
    * @param stepSize
    * @return
    */
  def trainWithParams(value: RDD[LabeledPoint], regParam: Double, numIterations: Int, stepSize: Double): LogisticRegressionModel = {

    //逻辑回归模型
    val lr = new LogisticRegressionWithSGD()
    //设置参数
    lr.optimizer
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .setStepSize(stepSize)

    //训练模型
    lr.run(value)
  }

  /**
    * 辅助函数,根据输入数据和分类模型,计算相关的指标
    *
    * @param str
    * @param data
    * @param model
    */
  def createMetrics(str: String, data: RDD[LabeledPoint], model: LogisticRegressionModel) = {
    //预测值
    val scoreAdnLabels = data.map {
      lp =>
        (model.predict(lp.features), lp.label)
    }
    //评估预测值
    val metrics = new BinaryClassificationMetrics(scoreAdnLabels)
    (str, metrics.areaUnderPR(), metrics.areaUnderROC())
  }


  def main(args: Array[String]): Unit = {
    sparkOperation(args)(modelTrain)
  }

}





































