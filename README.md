# HON-401----Spring-2025
Res: Deep Learning Algorithms

## Introduction

My name is Don Li, and I'm a student at Portland State University studying computer science and mathematics; I am also a member of the University Honors College (UHC) at Portland State. This project is part of my undergraduate research experience at Portland State through taking [HON 401](https://www.pdx.edu/honors/hon-401-research). My HON 401 research experience this term is focused on the algorithms that underlie neural networks, and I am doing this work under the supervision of [Prof. Dacian Daescu](https://web.pdx.edu/~daescu/). My research experience is centered on reading and reproducing the results of [Higham(2019)](https://arxiv.org/abs/1801.05894). 

Here are the contents of this readme file:

1. An Overview of Artificial Neural Networks (ANN's)
2. Cost Function Optimization for Neural Networks -- Stochastic Gradient & Backpropagation
3. Building & Training a Neural Network -- Iris Dataset Multi-Class Classification
4. Convolutional Neural Networks (CNN's) for Image Classification -- CIFAR-10 Dataset
5. Application -- A Neural Network for Binary Classification (Johns Hopkins Diabetes Dataset)

## An Overview of Artificial Neural Networks (ANN's)

In 1955, computer scientist John McCarthy [defined](https://hai-production.s3.amazonaws.com/files/2020-09/AI-Definitions-HAI.pdf) artificial intelligence as "the science and engineering of making intelligent machines". _Machine learning_ (ML) is a major subset of AI concerned with building AI models that can learn and make inferences from data. More precisely, machine learning is concerned with making predictions about a _target variable_ (i.e., the output of a model) using data from _feature variables_ (i.e., the inputs of a model). Machine learning models are concerned with either _regression_ tasks (i.e., predicting the value of a continuous variable) or _classification_ tasks (i.e., predicting a category that a data point belongs to). 

A major subset of machine learning is _deep learning_ (DL). Deep learning is concerned with performing either regression or classification tasks through a specific model type called an _(artificial) neural network_. Such artificial neural networks mimic the firing of neurons in the (human) brain to perform ML tasks. As Higham (2019) note, neural networks are widely applied to difficult computational tasks such as image recognition, speech recognition, and natural language processing (NLP). 

A neural network consists of _nodes_ that are connected to one another in _layers_ (in more theoretical terms, a neural network can be modeled as a directed graph). A neural network must consist of an _input layer_ (which consist of nodes that contain the neural network's inputs) and an _output layer_ (which yields the prediction made by a neural network). 


