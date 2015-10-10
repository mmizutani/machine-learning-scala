package com.example

object SupportVectorMachine {

  case class Data(x: Double, y: Double, t: Int) {
    var l = 0.0 // Lagrange multiple
  }

  class SVM(samples: Seq[Double]) {
    var (a, b, c) = (0.0, 0.0, 0.0)
    def predict(x: Double, y: Double): Boolean = a*x + b*y + c > 0

    def newc(): Double = {
      val pos = samples.filter(_.t == +1).map(d => a * d.x * b * d.y).min
      val neg = samples.filter(_.t == -1).map(d => a * d.x * b * d.y).min
      (pos + neg) / 2
    }
    def newa = (for(d <- samples) yield d.l * d.t * d.x).sum
    def newb = (for(d <- samples) yield d.l * d.t * d.y).sum
    def kkt(d: Data) = d.l match {
      case 0 => d.t * (a * d.x + b * d.y + c) >= 1
      case _ => d.t * (a * d.x + b * d.y + c) == 1
    }
  }

  def main(args: Array[String]): Unit = {

  }

}