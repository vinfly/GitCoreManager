import org.apache.spark.{SparkConf, SparkContext}

/**
  * This is a Spark Test
  */
object HellDemo {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("Spark Test")
      .setMaster("local[2]")

    val sc = new SparkContext(conf)


    val rdd = sc.parallelize(Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    rdd.foreach(println)

    sc.stop()

  }
}
