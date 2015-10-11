package com.example

object SupportVectorMachine {

  val samples = Seq(
    Data(-1.0, -0.5, -1),
    Data(-0.5, -1.0, -1),
    Data(-0.65, -0.65, -1),
    Data(-1.0, -1.5, -1),
    Data(-1.65, -1.65, -1),
    Data(0.0, 0.0, 1),
    Data(0.5, 0.0, 1),
    Data(0.2, 0.3, 1)
  )

  case class Data(x: Double, y: Double, t: Int) {
    var l = 0.0 // Lagrange multiple
  }

  class SVM(samples: Seq[Data]) {
    var (a, b, c) = (0.0, 0.0, 0.0)

    def predict(x: Double, y: Double): Boolean = a*x + b*y + c > 0

    private def newa = (for(d <- samples) yield d.l * d.t * d.x).sum
    private def newb = (for(d <- samples) yield d.l * d.t * d.y).sum
    private def newc: Double = {
      val pos = samples.filter(_.t == +1).map(d => a * d.x * b * d.y).min
      val neg = samples.filter(_.t == -1).map(d => a * d.x * b * d.y).max
      - (pos + neg) / 2
    }

    private def kkt(d: Data): Boolean = d.l match {
      case 0 => d.t * (a * d.x + b * d.y + c) >= 1
      case _ => d.t * (a * d.x + b * d.y + c) == 1
    }

    // update lambdai1 and lambdai2
    private def update(d1: Data, d2: Data) = {
      val (dx, dy) = (d1.x - d2.x, d1.y - d2.y)
      val den = dx * dx + dy * dy
      val num = (for(d <- samples)
        yield d.l * d.t * (dx * d.x + dy * d.y)
      ).sum - d1.t + d2.t
      val di1 = clip(d1, d2, -d1.t * num /den)
      d1.l += di1
      d2.l -= di1 * d1.t * d2.t
    }

    private def clip(d1: Data, d2: Data, di1: Double): Double = {
      if(d1.t == d2.t) {
        if(di1 < -d1.l) return -d1.l
        if(di1 > +d2.l) return +d2.l
        di1
      } else {
        Seq(-d1.l, -d2.l, di1).max
      }
    }

    var err = 1.0

    while (samples.exists(!kkt(_)) && err > 1e-7) {
      val i1 = samples.indexWhere(!kkt(_))
      var i2 = 0
      do i2 = (math.random * samples.size).toInt while (i1 == i2)
      println(s"i1=$i1, i2=$i2")
      update(samples(i1), samples(i2))
      val (olda, oldb, oldc) = (a, b, c)
      a = newa
      b = newb
      c = newc
      println(s"a=$a, b=$b, c=$c")
      err = (a-olda)*(a-olda) + (b-oldb)*(b-oldb) + (c-oldc)*(c-oldc)
      println(s"err = ${"%10.9f" format err}")
    }

    def printResult() = {
      println(s"Support vector found:\n a*x + b*y + c = $a*x + $b*y + $c = 0")
    }

  }

  def main(args: Array[String]): Unit = {
    val svm = new SVM(samples)
    for(i <- -2.0 to 2.0 by 0.2; j <- -2.0 to 2.0 by 0.2) {
      println(s"(${"%4.3f" format i}, ${"%4.3f" format j}): ${svm.predict(i, j)}")
    }
    svm.printResult()
  }

}