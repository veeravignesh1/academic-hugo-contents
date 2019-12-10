---
title: Machine Learning - Andrew Ng
author: Veera Vignesh
date: '2019-12-10'
slug: Machine-Learning
categories:
  - Machine Learning
tags:
  - Octave
  - Python
subtitle: ''
summary: 'Course notes of Machine Learning by Andrew Ng'
authors: []
lastmod: '2019-12-10T11:22:37+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

**Main aim of this course is to understand,implement & interpret Machine Learning algorithms**

## [Week 1](Week1.pdf)

## What is Machine Learning

> ​	The Field of study that gives computes the ability to learn without being explicitly programmed to do so. - Arthur Samuel

> ​	Computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. - Tom Mitchell

<u>Example</u> - For a machine to classify the email as spam or not-spam. Task (T) is the process of classifying the mail is spam or not, Experience(E) is the number of emails it gets to classify, Performance(P) is the ability of the program to classify the mail accurately.

Machine Learning Types:

- **Supervised** 
- **Unsupervised**
- Reinforcement
- Recommender systems

**Supervised Machine Learning Algorithm:**

We have an idea that there is a relation between our input and output variable.  Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. 

<u>Example</u> - House Price Prediction is supervised because the right answer or actual price of the house is given during training the model. This kind of problem is commonly known as **Regression Problem** - Predicts the continuous output.

<u>Example</u> - Classification of Cancer as Malignant (Harmful) or Benign (Harmless). Identifying the probability of the given observation being malignant or benign- This kind of problem is commonly known as **Classification Problem** - Based on the probability prediction is done.

Classification problem can also classify with multiple levels on the target variable (not limited to binary class of the target variable)

**Unsupervised Learning**

Allowing the machine to identify the patterns in the given data. There is no labels given to the data (no target variable).  Problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables. 

<u>Example</u> 

1. Google News - Groups them into cohesive groups (clusters)
2. Genome - Identifying the type of people based on the structure of the genome
3. Organize computer clusters
4. Social Network analysis
5. Market segmentation
6. Astronomical data analysis
7. [Cocktail Party algorithm](https://en.wikipedia.org/wiki/Cocktail_party_effect) -[ Extracting voice from musical mixtures](https://arxiv.org/abs/1504.04658)











