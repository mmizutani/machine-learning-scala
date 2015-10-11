package com.example

import scala.collection.mutable.{ Map, Set }
import net.java.sen.SenFactory
import net.java.sen.dictionary.Token

import scala.io.Source

object NaiveBayesianClassifier {

  case class Label(id: String)
  case class Data(words: Seq[String], label: Label)

  class NaiveBayes(samples: Seq[Data]) {
    val labels = Set[Label]()
    val vocab = Set[String]() // set of different words in the training data set
    val prior = Map[Label, Int]() // counts prior distributions
    val denom = Map[Label, Int]() // counts number of words with replacement in each class
    val numer = Map[(String, Label), Int]() // counts sets of words and classes

    /**
     * Conditional probability $$ P(w|C) $$ with Laplace smoothing
     * @param w word
     * @param c class
     * @return probability
     */
    def pwc(w: String, c: Label): Double = numer.get((w, c)) match {
      case Some(num) => (num + 1.0) / (denom(c) + vocab.size)
      case None      =>        1.0  / (denom(c) + vocab.size)
    }

    /**
     * Calculates the likelihood times prior probability $$ P(D|C)P(C) $$
     * @param words
     * @param label
     * @return
     */
    def pdc(words: Seq[String], label: Label): Double = {
      val pc = Math.log(prior(label).toDouble / samples.size)
      pc + words.map(w => math.log(pwc(w, label))).sum
    }

    /**
     * Identifies the class which maximizes the posterior probability $$ P(C|D) $$
     * @param words
     * @return identified label
     */
    def predict(words: Seq[String]): Label = labels.maxBy(pdc(words, _))

    samples.foreach { case Data(words, label) =>
      labels += label
      words.foreach(w => numer((w, label)) = 0)
    }
    labels.foreach(prior(_) = 0)
    labels.foreach(denom(_) = 0)
    samples.foreach { case Data(words, label) =>
      vocab ++= words
      prior(label) += 1
      words.foreach(w => denom(label) += 1)
      words.foreach(w => numer((w, label)) += 1)
    }

  }

  def main(args: Array[String]): Unit = {
    val tagger = SenFactory.getStringTagger(null)
    val tokens = new java.util.ArrayList[Token]()
    val source = Source.fromFile("README.md")
    tagger.analyze(source.mkString, tokens)
    import collection.JavaConversions._
    for (morph <- tokens.map(_.getMorpheme)) {
      val ps = morph.getPartOfSpeech
      val bf = morph.getBasicForm
      println(ps)
    }
  }

}