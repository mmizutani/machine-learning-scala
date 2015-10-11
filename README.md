# Scala で実装する機械学習入門
================================
>Introduction to major machine learning algorithms in Scala

>http://pafelog.net/mining.pdf

## Scala demos of major machine-learning algorithms

1. [Nearest Neighborhood](src/main/scala/com/example/NearestNeighborhood.scala)
2. [Neural Network](src/main/scala/com/example/NeuralNetwork.scala)
3. [Support Vector Machine](src/main/scala/com/example/SupportVectorMachine.scala)
4. [Support Vector Machine with Kernel Trick](src/main/scala/com/example/SVMWithKernelTrick.scala)
5. [Naive Bayesian Classifier](src/main/scala/com/example/NaiveBayesianClassifier.scala)
6. [Maximum Likelihood](src/main/scala/com/example/MaximumLikelihood.scala)


## How to run the examples above

```
$ git clone git@github.com:mmizutani/machine-learning-scala.git
$ activator run
...
Multiple main classes detected, select one to run:

 [1] com.example.MaximumLikelihood
 [2] com.example.NaiveBayesianClassifier
 [3] com.example.NearestNeighborhood
 [4] com.example.NeuralNetwork
 [5] com.example.NeuralNetworkMultiLayer
 [6] com.example.SVMWithKernelTrick
 [7] com.example.SupportVectorMachine

Enter number:
```