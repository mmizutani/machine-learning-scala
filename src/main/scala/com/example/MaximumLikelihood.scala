package com.example

import org.apache.commons.math3.distribution.NormalDistribution

import scala.util.Random

object MaximumLikelihood {

  /**
   * Expectation maximizer
   * @param samples
   * @param K number of parameters to estimate
   */
  class EM(samples: Seq[Double], K: Int) {
    val P = Array.ofDim[Double](K, samples.size)
    val W = Array.ofDim[Double](K)
    val M = Array.ofDim[Double](K)
    val S = Array.ofDim[Double](K)

    val denom = 1 / math.sqrt(2 * Math.PI)
    def normal(x: Double, m: Double, s: Double) = {
      val pow = 0.5 * (x - m) * (x - m) / s
      denom / math.sqrt(s) * math.exp(- pow)
    }

    for (k <- 0 until K) {
      W(k) = math.random
      M(k) = math.random
      S(k) = math.random
    }
    val wsum = W.sum
    for (k <- 0 until K) W(k) /= wsum

    var logL = 0.0
    for (step <- 1 to 100) {
      for ((x, i) <- samples.zipWithIndex) {
        for (k <- 0 until K) P(k)(i) = W(k) * normal(x, M(k), S(k))
        val sum = (for (k <- 0 until K) yield P(k)(i)).sum
        for (k <- 0 until K) P(k)(i) = P(k)(i) / sum
      }
      for (k <- 0 until K) {
        val nk = P(k).sum
        M(k) = (P(k) zip samples).map{case(p, x) => p * x * 1}.sum / nk
        S(k) = (P(k) zip samples).map{case(p, x) => p * x * x}.sum / nk
        S(k) = S(k) - M(k) * M(k)
      }
    }
  }

  def main(args: Array[String]): Unit = {
    val w = Seq(0.57454, 0.42546)
    val mean = Seq(0.89739, 0.19543)
    val sd = Seq(0.69749, 0.83826)
    var samples: Seq[Double] = Seq()
    for (i <- 0 to 1) {
      val generator = new NormalDistribution(mean(i), sd(i))
      samples ++= Seq.fill(10000)(w(i) * generator.inverseCumulativeProbability(Random.nextDouble()))
    }
    //println(samples)

    val em = new EM(samples, 2)
    println(s"MLE Result:\n W:${em.W.deep}\n M:${em.M.deep}\n S:${em.S.deep}")
  }

}