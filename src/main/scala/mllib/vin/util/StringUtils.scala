package mllib.vin.util

object StringUtils {

  def trimChar(str: String): String = {

    if (str.endsWith("-")) str.substring(0, str.length - 1) else str

  }

}
