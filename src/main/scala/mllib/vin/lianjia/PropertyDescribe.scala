package mllib.vin.lianjia

import com.hankcs.hanlp.HanLP
import com.hankcs.hanlp.tokenizer.{NLPTokenizer, StandardTokenizer}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.{HashingTF, IDF, Word2Vec}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object PropertyDescribe {

  def sparkOperation(args: Array[String])(operation: SparkSession => Unit): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .appName("PropertyDescribe")
      .master("local[4]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try {
      operation(spark)
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      spark.stop()
    }
  }

  def modelTrain(spark: SparkSession): Unit = {
    val sc = spark.sparkContext

    val newsRDD: RDD[String] = sc.textFile("/usr/spark/propertydescribe/*")

    println(s"Count: ${newsRDD.count()}")
    //println(s"First: \n ${newsRDD.first()}")
    println(s"去除NULL值 Count: ${newsRDD.filter(_.trim != "NULL").count()}")

    //过滤NULL值
    val filteredRDD = newsRDD.filter(line => {
      val array = line.split("\t")
      !array(1).equals("\\N") && !array(1).equals("NULL") && !array(1).trim.equals("")
    })
    println(s"filtered Count: ${filteredRDD.count()}")

    //分词
    //1.HanLP
    val hanLPRDD = filteredRDD.map(line => {
      val describe = line.split("\t")(1)
      val features = HanLP.segment(describe.toString)
        .toArray
        .mkString("\t")
        .replaceAll("[，,/?#$@*。;:\"、.！】【 ：）（=]", "")
        .replaceAll("[a-zA-Z]", "")
      (features, line.split("\t")(2))
    })
    println("HanLP : " + hanLPRDD.take(10).foreach(println))


    //提取特征和标签 组成二元组
    val labelfeaturesRDD = hanLPRDD.map { tuple =>
      //将分词后的汉字组成数组
      val words: Array[String] = tuple._1.split("\t")
      //将成交类别数据转化为0(未成交)和1(成交)
      val category: Double = tuple._2 match {
        case "\\N" => 0.0
        case _ => 1.0
      }
      //返回二元组(category ,title)
      (category, words)
    }

    //    //TODO: 从分词后的单词中提取特征值(转换为数值类型数据)
    /**
      * 词袋模型:BOW
      * 在SparkMLlib中提供TF和IDF类
      */
    //创建HashTF对象,用于生成词频向量 class HashingTF(val numFeatures:Int)
    val hashTF = new HashingTF(10000)

    //提取特征值,已经将文本转换为向量
    val lpsRDD: RDD[LabeledPoint] = labelfeaturesRDD.map {
      case (category, words) =>

        //转换单词为向量
        val tfVector: linalg.Vector = hashTF.transform(words)

        //返回
        LabeledPoint(category, tfVector)
    }

    //获取IDF和TF的值
    val tfVectorRDD: RDD[linalg.Vector] = lpsRDD.map(_.features)

    //IDF Model ,用于计算IDF值
    val idfModel = new IDF().fit(tfVectorRDD)

    //给单词加权重(权重 TF-IDF)
    val lpsTfIdfRDD: RDD[LabeledPoint] = lpsRDD.map {
      case LabeledPoint(label, features) => LabeledPoint(label, idfModel.transform(features))
    }

    //    lpsTfIdfRDD.take(5).foreach(println)

    //TODO:交叉验证:训练数据和测试数据集
    val Array(trainingRDD, testingRDD) = lpsTfIdfRDD.randomSplit(Array(0.8, 0.2), seed = 123L)

    //TODO: 使用朴素贝叶斯算法模型训练数据集
    val nbModel: NaiveBayesModel = NaiveBayes.train(trainingRDD)

    //使用模型进行预测
    val nbPredictRDD: RDD[(Double, Double)] = testingRDD.map(lp => (lp.label, nbModel.predict(lp.features)))

    //获取前十条结果
    //     nbPredictRDD.take(10).foreach(println)

    /**
      * TODO : 评估预测的值
      */
    def evaluateNbModel(predictRDD: RDD[(Double, Double)]): Unit = {

      //          //统计预测值占比
      //          val compareRDD = predictRDD.map{
      //            case(actural,predict)=>(actural.round-predict.round).toInt
      //          }
      //          val equalsCount = compareRDD.filter(_ == 0).count()
      //          println(s"总的测试数据集个数为: ${compareRDD.count()},相等的个数为: ${equalsCount}")
      //
      //          //计算MSE: 均方误差
      //          val nbMSE = predictRDD.map{
      //            case(actural,predict) => Math.pow(actural-predict,2)
      //          }.reduce(_ + _)/predictRDD.count()
      //
      //          println(s"均方误差MSE: $nbMSE")
      //          println(s"均方根误差RMSE: ${Math.sqrt(nbMSE)}")

      //统计实际成交的个数
      val acturalDealCount = predictRDD.filter(_._1.toInt == 1).count()
      //统计实际成交,预测也成交的个数
      val predictDealCount = predictRDD.filter(_._1.toInt == 1).filter(_._2.toInt == 1).count()
      println(s"实际成交的个数: ${acturalDealCount}")
      println(s"实际成交,预测也成交的个数: ${predictDealCount}")

    }

    evaluateNbModel(nbPredictRDD)
    //
    //    /**
    //      * TODO : 模型参数调优
    //      * 朴素贝叶斯设置参数优化性能
    //      */
    //    def trainNbWithParams(input:RDD[LabeledPoint],lambda:Double):NaiveBayesModel={
    //      //构建对象实例
    //      val nb = new NaiveBayes()
    //      //设置lambda
    //      nb.setLambda(lambda)
    //      nb.run(input)
    //    }
    //    def testNbWithParams(trainRDD:RDD[LabeledPoint],testRDD:RDD[LabeledPoint]):Unit={
    //      //不同的参数
    //      val nbResults:Seq[(String,Double)] = Seq(0.0001,0.001,0.01,0.1,1.0,10.0).map{ param =>
    //
    //        val model = trainNbWithParams(trainRDD,param)
    //        val scoreAndLabels = testRDD.map(lp => (lp.label,model.predict(lp.features)))
    //        val metrcis = new BinaryClassificationMetrics(scoreAndLabels)
    //        (s"$param lambda",metrcis.areaUnderROC())
    //      }
    //
    //      nbResults.foreach{
    //        case(param,auc)=> println(s"$param, AUC = ${auc * 100}%2.2f%%")
    //      }
    //    }
    //    testNbWithParams(trainingRDD,testingRDD)
    //
    //    //TODO: 将模型持久化
    //    //nbModel.save(sc,"datas/classification/newsdata/model")
    //
    //    //根据已有的模型进行预测,实际值e
    //    val title = "Mariah Carey's jealousy reportedly partly to blame for her split from Nick Cannon"
    //    //提取特征
    //    val feature = idfModel.transform(
    //      hashTF.transform(
    //        title.split(" ").map(word => word.toLowerCase.replace(",","").replace("'",""))
    //      )
    //    )
    //
    //    //调用模型进行预测
    //    val predictCategory:String = nbModel
    //      .predict(feature) match {
    //      case 0.0 => "business"
    //      case 1.0 => "science and technology"
    //      case 2.0 => "entertainment"
    //      case 3.0 => "health"
    //      case _ => "unknown"
    //    }
    //    println(s"Predict Category: $predictCategory")
    //
    //    /**
    //      * TODO: 获取相近的单词
    //      */
    //
    //    val inputRDD:RDD[Seq[String]] = titleCategoryRDD.map{
    //      case(category,title) =>
    //        //对新闻标题进行分词
    //        val words:Array[String] = title
    //          .split(" ")
    //          .map(word=>word.toLowerCase.replace(",","").replace("'",""))
    //        words.toSeq
    //    }
    //    val word2Vec = new Word2Vec()
    //    val word2VecModel = word2Vec.fit(inputRDD)
    //    //获取某个单词前10个相近的词汇
    //    val synonyms:Array[(String,Double)] = word2VecModel.findSynonyms("STOCKS".toLowerCase,10)
    //    synonyms.foreach(println)

  }

  def main(args: Array[String]): Unit = {
    sparkOperation(args)(modelTrain)
  }
}






