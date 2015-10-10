package com.example

object NearestNeighborhood {

  case class Data(point: Seq[Double], answer: Any = 0)

  val samples = Seq(
    Data(Seq(-0.10, +0.50), 1),
    Data(Seq(-0.45,+1.30), 1),
    Data(Seq(+0.60,+0.75), 1),
    Data(Seq(+0.30,+1.25), 1),
    Data(Seq(+0.75,+0.70), 1),
    Data(Seq(+1.20,+0.55), 1),
    Data(Seq(+0.20,+0.30), 0),
    Data(Seq(+1.60,+0.60), 0),
    Data(Seq(+0.40,+0.55), 0),
    Data(Seq(+0.60,+0.40), 0),
    Data(Seq(+0.80,+0.55), 0),
    Data(Seq(+1.25,+0.20), 0)
  )

  class KNN(k: Int, samples: Seq[Data], distance: (Data, Data) => Double) {
    def predict(target: Data): Any = {
      val knears = samples.sortWith((t1, t2) => {
        val d1 = distance(t1, target)
        val d2 = distance(t2, target)
        d1 < d2
      }).take(k).map(_.answer)
      val answers = knears.toSet
      val verdict = answers.map(ans => ans -> knears.count(_ == ans))
      verdict.maxBy(tuple => tuple._2)._1
    }
  }

  val euclid = (a: Data, b: Data) => Math.sqrt(
    Math.pow(a.point(0) - b.point(0), 2) +
    Math.pow(a.point(1) - b.point(1), 2)
  )

  def main(args: Array[String]): Unit = {
    val knn = new KNN(3, samples, euclid)
    println(knn.predict(Data(Seq(0.0, 1.0))))
    println(knn.predict(Data(Seq(1.0, 0.0))))
    println(knn.predict(Data(Seq(0.3, 0.2))))
    println(knn.predict(Data(Seq(0.7, 0.6))))
    println(knn.predict(Data(Seq(0.99, 0.0))))
    println(knn.predict(Data(Seq(0.0, 0.99))))
  }

}
