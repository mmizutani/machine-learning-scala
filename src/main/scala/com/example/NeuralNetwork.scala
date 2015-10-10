package com.example

object NeuralNetwork {

  class Perceptron(samples: Seq[(Seq[Double], Boolean)], dim: Int) {
    val weights = new Array[Double](dim + 1)
    def predict(in: Seq[Double]): Boolean = ((1.0 +: in) zip weights).map {
      case (i, w) => i * w
    }.sum >= 0
    val eta = 0.1
    for(step <- 1 to 10000) samples.foreach { case (in, out) =>
      predict(in) match {
        case true if !out =>
          for(k <- 0 until dim) weights(k+1) -= eta * in(k)
          weights(0) -= eta * 1
        case false if out =>
          for(k <- 0 until dim) weights(k+1) += eta * in(k)
          weights(0) += eta * 1
        case _ =>
      }
    }
  }

  val samples = Seq(
    (Seq(0.0, 0.0), false),
    (Seq(0.0, 1.0), true),
    (Seq(1.0, 0.0), true),
    (Seq(1.0, 1.0), true)
  )

  def main(args: Array[String]): Unit = {
    val p = new Perceptron(samples, 2)
    println(p.predict(Seq(0.0, 0.0))) // false
    println(p.predict(Seq(0.0, 1.0))) // true
    println(p.predict(Seq(1.0, 0.0))) // true
    println(p.predict(Seq(1.0, 1.0))) // true

    println(p.predict(Seq(0.1, 0.8))) // false
    println(p.predict(Seq(0.1, 0.9))) // true
    println(p.predict(Seq(0.2, 0.9))) // true
    println(p.predict(Seq(0.05, 0.9))) // false

    println(p.weights.mkString(",")) // -0.1,0.1,0.1  -0.1x+0.1y+0.1=0
  }

}