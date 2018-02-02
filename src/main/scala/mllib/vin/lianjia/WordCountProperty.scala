package mllib.vin.lianjia

import com.hankcs.hanlp.HanLP
import mllib.vin.util.StringUtils
import org.apache.spark.{SparkConf, SparkContext}


object WordCountProperty {
  /**
    * 贷出模式： 贷出函数，在Scala语言中涉及到资源释放
    *
    * @param args
    * 传递参数，从MAIN传递过来
    * @param operation
    * 贷出模式中的 用户函数，真正数据的地方
    */
  def sparkOperation(args: Array[String])(operation: SparkContext => Unit): Unit = {

    // 构建 SparkConf实例，设置应用的配置信息
    val sparkConf = new SparkConf()
      .setAppName("WordCountProperty Application")
      .setMaster("local[4]")
    // 创建SparkContext实例，读取要处理的数据和调度Jobs
    val sc = SparkContext.getOrCreate(sparkConf)
    // 设置日志级别信息
    sc.setLogLevel("WARN")
    try {
      // 调用 用户函数处理数据
      operation(sc)
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      // 关闭资源
      sc.stop()
    }
  }


  /**
    * 贷出模式：用户函数，业务逻辑  编程实现
    * 此处主要在于 数据的转换，模型的训练，测试，验证和调优
    *
    * @param sc
    * SparkContext 实例对象
    */
  def modelTrain(sc: SparkContext): Unit = {

    val sourceRDD = sc.textFile("/usr/spark/allproperty/*")

    //将第二列null值,空值,\N,"无"统一变为NULL,第三列时间戳变为1,\N变为0
    val processedRDD = sourceRDD.map {
      case line =>

        val arrayLine = if (line.split("\t").length == 3) line.split("\t") else new Array[String](3)

        if (arrayLine(1).trim.equals("无") ||
          arrayLine(1).trim.equals("\\N") ||
          arrayLine(1).trim.equals("") ||
          arrayLine(1).trim.equals("无要求")
        ) {
          arrayLine(1) = "NULL"
        }
        if (arrayLine(2).trim.equals("\\N")) {
          arrayLine(2) = 0.toString
        } else {
          arrayLine(2) = 1.toString
        }
        arrayLine
    }

    //分词,过滤
    val filteredRDD = processedRDD.map(array => {
      val words = HanLP.segment(array(1)).toArray
        .mkString("\t")
        .replaceAll("[，,\\/?#$@*。;:\"、.！】【 ：）（=!-]", "")
        .replaceAll("[a-z]", "")
      (words, array(2))
    })

    //只留下成交的房源的房源描述
    val filteredDealRDD = filteredRDD.filter(_._2.toInt == 1).map(_._1)


    val wordCountRDD = filteredDealRDD.flatMap(_.split("\t")).map((_, 1)).reduceByKey(_ + _).sortBy(_._2, false)


    //wordCountRDD.take(100).foreach(println)

    //所有词频统计
    val allwordCountRDD = filteredRDD.map(_._1).flatMap(_.split("\t")).map((_, 1)).reduceByKey(_ + _).sortBy(_._2, false)

    // allwordCountRDD.take(100).foreach(println)


    val joinRDD = wordCountRDD.join(allwordCountRDD)


    val sortedRDD = joinRDD.map(tuple => {
      val keyWords = tuple._1

      val dealNums = tuple._2._1
      val allNums = tuple._2._2
      val percent = (dealNums.toDouble / allNums.toDouble) * 100

      val sortKey = new SortKey(dealNums.toLong, allNums.toLong, percent.toLong)
      Tuple2(sortKey, (keyWords, dealNums, allNums, percent))
    }).sortByKey(false)

    //    for (tuple <- sortedRDD.collect()) {
    //      println(tuple._2)
    //    }
    //


    //  filteredRDD.foreach(println)

    // 组合
    val aggRDD = filteredRDD.map(tuple => {
     // val keyWordsArray = Array("正常", "首付", "置换", "[0~9]")
     val keyWordsArray = Array("正常", "即可", "接受", "可以", "越多", "越好", "没有", "无", "要求", "首付", "置换", "付款", "成")
      // val keyWordsArray = Array("要求", "首付", "接受", "")

      val stringBuffer = new StringBuffer("")
      for (e <- keyWordsArray) {
        if (tuple._1.contains(e)) {
          stringBuffer.append(e).append("-")
        }
      }

      (StringUtils.trimChar(stringBuffer.toString), tuple._1, tuple._2)
    })

    //计算组合后的词频
    //只计算交易的
    val dealAggRDD = aggRDD.filter(_._3.toInt == 1).map(tuple => {
      (tuple._1, 1)
    }).reduceByKey(_ + _).sortBy(_._2, false)

    val allAggRDD = aggRDD.map(tuple => {
      (tuple._1, 1)
    }).reduceByKey(_ + _).sortBy(_._2, false)


    val joinAggRDD = dealAggRDD.join(allAggRDD)

    val sortedAggRDD = joinAggRDD.map(tuple => {
      val keyWords = tuple._1

      val dealNums = tuple._2._1
      val allNums = tuple._2._2
      val percent = (dealNums.toDouble / allNums.toDouble) * 100

      val sortKey = new SortKey(dealNums.toLong, allNums.toLong, percent.toLong)
      Tuple2(sortKey, (keyWords, dealNums, allNums, percent))

    }).sortByKey(false)

        for (tuple <- sortedAggRDD.collect()) {
          println(tuple._2)
        }


  }

  /**
    * Spark Application 程序入口
    *
    * @param args
    * 程序传递的参数
    */
  def main(args: Array[String]): Unit = {
    // 调用的 贷出函数，将用户函数作为参数传递
    sparkOperation(args)(modelTrain)
  }

}
