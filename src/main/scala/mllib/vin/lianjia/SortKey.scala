package mllib.vin.lianjia

class SortKey(val dealCounts: Long, val allCounts: Long, val percentCunt: Long) extends Ordered[SortKey] with Serializable {
  def compare(that: SortKey): Int = {
    if (dealCounts - that.dealCounts != 0) {
      (dealCounts - that.dealCounts).toInt
    } else if (allCounts - that.allCounts != 0) {
      (allCounts - that.allCounts).toInt
    } else if (percentCunt - that.percentCunt != 0) {
      (percentCunt - that.percentCunt).toInt
    } else {
      0
    }
  }

}