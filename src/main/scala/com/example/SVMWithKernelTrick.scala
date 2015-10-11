package com.example

import org.jfree.chart.{ChartFrame, ChartFactory}
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.{XYSeriesCollection, XYSeries}

object SVMWithKernelTrick {

  val samples = Seq(
    Data(Seq(-1.0, -0.5), -1),
    Data(Seq(-0.5, -1.0), -1),
    Data(Seq(-0.65, -0.65), -1),
    Data(Seq(-1.0, -1.5), -1),
    Data(Seq(-1.65, -1.65), -1),
    Data(Seq(0.0, 0.0), 1),
    Data(Seq(0.5, 0.0), 1),
    Data(Seq(0.2, 0.3), 1)
  )

  case class Data(p: Seq[Double], t: Int) {
    var l = 0.0 // Lagrange multiple
    def -(that: Data): Data = {
      val newp: Seq[Double] = (this.p zip that.p).map { case (p1, p2) => p1 - p2 }
      //http://stackoverflow.com/questions/1991240/scala-method-operator-overloading
      new Data(newp, 1)
    }
  }

  class SVM(samples: Seq[Data], k: (Data, Data) => Double) {
    val L = new Array[Double](samples.size) // undetermined multiples
    val C = 1e100 // soft-margin multiple
    var c = 0.0 // constant

    def wx(x: Data) = (for(s <- samples) yield s.l * s.t * k(x, s)).sum

    private def kkt(d: Data): Boolean = d.l match {
      case 0 => d.t * (wx(d) * c) >= 1
      case C => d.t * (wx(d) * c) <= 1
      case _ => d.t * (wx(d) * c) == 1
    }

    // update lambdai1 and lambdai2
    private def update(d1: Data, d2: Data) = {
      val den = k(d1, d1) - 2 * k(d1, d2) + k(d2, d2)
      val num = wx(d1 - d2) - d1.t + d2.t
      val di1 = clip(d1, d2, -d1.t * num /den)
      d1.l += di1
      d2.l -= di1 * d1.t * d2.t
    }

    private def newc: Double = {
      val pos = samples.filter(_.t == +1).map(wx(_)).min
      val neg = samples.filter(_.t == -1).map(wx(_)).max
      (pos + neg) / 2
    }

    private def clip(d1: Data, d2: Data, di1: Double): Double = {
      if(d1.t == d2.t) {
        val L = math.max(-d1.l, d2.l - C)
        val H = math.min(+d2.l, C - d1.l)
        if(di1 < L) return L
        if(di1 > H) return H
        di1
      } else {
        val L = math.max(-d1.l, -d2.l)
        val H = math.min(-d1.l, -d2.l) + C
        if(di1 < L) return L
        if(di1 > H) return H
        di1
      }
    }

    def predict(p: Data): Boolean = wx(p) + c > 0

    var err = 1.0

    while (samples.exists(!kkt(_)) && err > 1e-7) {
      val i1 = samples.indexWhere(!kkt(_))
      var i2 = 0
      do i2 = (math.random * samples.size).toInt while (i1 == i2)
      //println(s"i1=$i1, i2=$i2")
      update(samples(i1), samples(i2))
      val oldc = c
      c = newc
      err = (c-oldc)*(c-oldc)
      println(s"err = ${"%10.9f" format err}")
    }

    def printResult() = {
      println(s"Support vector found:\n w*x + c =  + $c = 0")
    }

  }

  object SVM {
    val gaussianKernel: (Data, Data) => Double = { (d1: Data, d2: Data) =>
      val sqEucDist = (d1.p zip d2.p).map { case (d1p, d2p) => math.pow(d1p - d2p, 2)}.sum
      val s = 1.0
      math.exp(- sqEucDist / (2 * s * s))
    }
  }

  def main(args: Array[String]): Unit = {
    val svm = new SVM(samples, SVM.gaussianKernel)
    val seriesA = new XYSeries("true data points")
    val seriesB = new XYSeries("false data points")
    for(i <- -2.0 to 2.0 by 0.2; j <- -2.0 to 2.0 by 0.2) {
      val d = Data(Seq(i, j), 1)
      println(s"(${"%4.3f" format i}, ${"%4.3f" format j}): ${svm.predict(d)}")
      if (i + j < 0) seriesA.add(i, j)
      else  seriesB.add(i, j)
    }
    svm.printResult()

    val dataset = new XYSeriesCollection()
    dataset.addSeries(seriesA)
    dataset.addSeries(seriesB)
    val chart = ChartFactory.createScatterPlot(
      "Dataset for Support Vector Machine",
      "x",
      "y",
      dataset,
      PlotOrientation.VERTICAL,
      true,
      true,
      false
    )
    val frame = new ChartFrame("Dataset", chart)
    frame.pack()
    frame.setVisible(true)
  }

}