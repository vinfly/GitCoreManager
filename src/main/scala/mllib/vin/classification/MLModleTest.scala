package mllib.vin.classification

import org.apache.spark.{SparkConf, SparkContext}


object MLModleTest {

  /**
    * 贷出模式： 贷出函数，在Scala语言中涉及到资源释放
    *
    * @param args
    * 传递参数，从MAIN传递过来
    * @param operation
    * 贷出模式中的 用户函数，真正数据的地方
    */
  def sparkOperation(args: Array[String])(operation: SparkContext => Unit): Unit = {
    // 如果通过参数传递
    if (args.length != 2) {
      println("Uasge: <AppName> <Master>")
      throw new IllegalArgumentException("Need Two Args...............")
    }

    // 构建 SparkConf实例，设置应用的配置信息
    val sparkConf = new SparkConf()
      .setAppName("MLClassification Application")
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
