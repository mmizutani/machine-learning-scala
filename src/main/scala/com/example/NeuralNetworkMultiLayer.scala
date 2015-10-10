package com.example

object NeuralNetworkMultiLayer {

  case class Data(input: Seq[Double], answer: Seq[Double])

  val samples = Seq(
    Data(Seq(0.0, 0.0), Seq(0.0)),
    Data(Seq(0.0, 1.0), Seq(1.0)),
    Data(Seq(1.0, 0.0), Seq(1.0)),
    Data(Seq(1.0, 1.0), Seq(0.0))
  )

  /**
   * Three-layer perceptron classifier
   * @param samples
   * @param I Dimension of input layer weights
   * @param H Dimension of intermediate hidden layer weights
   * @param O Dimension of output layer weights
   */
  class MLP(samples: Seq[Data], I: Int, H: Int, O: Int) {
    val w1 = scala.collection.mutable.Map[(Int, Int), Double]()
    val w2 = scala.collection.mutable.Map[(Int, Int), Double]()
    for(i <- 0 until I; h <- 0 until H)
      w1((i, h)) = 2 * Math.random -1
    for(h <- 0 until H; o <- 0 until O)
      w2((h, o)) = 2 * Math.random -1

    private val eta = 0.1

    private def sigmoid(x: Double) = 1.0 / (1.0 + Math.exp(-x))

    private def hidden(args: Seq[Double]): Seq[Double] = {
      val hid = new Array[Double](H)
      for(i <- 0 until I; h <- 0 until H)
        hid(h) += w1((i, h)) * args(i)
      for(h <- 0 until H)
        hid(h) = sigmoid(hid(h))
      hid
    }

    private def output(args: Seq[Double]): Seq[Double] = {
      val out = new Array[Double](O)
      for(h <- 0 until H; o <- 0 until O)
        out(o) += w2((h, o)) * args(h)
      for(o <- 0 until O)
        out(o) = sigmoid(out(o))
      out
    }

    def predict(input: Seq[Double]) = {
      val result = output(hidden(input)).head
      (result > 0.5, result)
    }

    for(step <- 1 to 10000) samples.foreach { case Data(input, ans) => {
      val hid = hidden(input)
      val out = output(hid)
      for(h <- 0 until H) {
        val e2 = for(o <- 0 until O) yield ans(o) - out(o)
        val g2 = for(o <- 0 until O) yield out(o) * (1 - out(o)) * e2(o)
        for(o <- 0 until O) w2((h, o)) += eta * g2(o) * hid(h)

        val e1 = for(o <- 0 until O) yield g2(o) * w2((h, o))
        val g1 = for(i <- 0 until I) yield hid(h) * (1 - hid(h)) * e1.sum
        for(i <- 0 until I) w1((i, h)) += eta * g1(i) * input(i)
      }
    }}

  }

  def main(args: Array[String]): Unit = {

    val p = new MLP(samples, 2, 5, 1)

    println(p.predict(Seq(0.0, 0.0))) // false
    println(p.predict(Seq(0.0, 1.0))) // true
    println(p.predict(Seq(1.0, 0.0))) // true
    println(p.predict(Seq(1.0, 1.0))) // true

    println(p.predict(Seq(0.1, 0.8))) // false
    println(p.predict(Seq(0.1, 0.9))) // true
    println(p.predict(Seq(0.2, 0.9))) // true
    println(p.predict(Seq(0.05, 0.9))) // false

    println(p.predict(Seq(1.5, -0.5)))
    println(p.predict(Seq(2.0, 0.0)))

    println(p.w1.mkString(",")) // -0.1,0.1,0.1  -0.1x+0.1y+0.1=0
    println(p.w2.mkString(",")) // -0.1,0.1,0.1  -0.1x+0.1y+0.1=0

  }

}