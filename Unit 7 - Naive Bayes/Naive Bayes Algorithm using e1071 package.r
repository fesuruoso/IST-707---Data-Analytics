'''
title: "Tutorial: naive Bayes in Pakcage e1071 for Titanic Prediction"
author: "Funmi Esuruoso"
date: "11/13/2019"

This is a tutorial on using the naive Bayes algorithm in the e1071 package to predict Titanic survivors.
https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
'''
# Load the required packages.
require(dplyr)
require(caret)
require(rpart)
require(e1071)
require(stringr)

# prepare data
#First load the training data in csv format, and then convert "Survived" to nominal variable and "Pclass" to ordinal variable.
library(RWeka)
trainset <- read.csv("F:/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Wk 7 - Naive Bayes/Data/titanic-train.csv")
trainset$Survived=factor(trainset$Survived)
trainset$Pclass=ordered(trainset$Pclass)

#Then load the test data and convert attributes in similar way.
testset <- read.csv("F:/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Wk 7 - Naive Bayes/Data/titanic-test.csv")
testset$Survived=factor(testset$Survived)
testset$Pclass=ordered(testset$Pclass)

#Then remove some attributes that are not likely to be helpful, such as "embarked" - create a new data set with all other attributes. Process the train and test set in the same way. 
myVars=c("Pclass", "Sex", "Age", "SibSp", "Fare", "Survived")
newtrain=trainset[myVars]
newtest=testset[myVars]

# naive Bayes in e1071
#Now load the package e1071
library(e1071)

#Build naive Bayes model using the e1071 package
nb=naiveBayes(Survived~., data = newtrain, laplace = 1, na.action = na.pass)

#Apply the model to predicting test data
pred=predict(nb, newdata=newtest, type=c("class"))

#Combine the predictions with the corresponding case ids. 
myids=c("PassengerId")
id_col=testset[myids]
newpred=cbind(id_col, pred)

#Add header to output
colnames(newpred)=c("Passengerid", "Survived")

#Write output to file
write.csv(newpred, file="F:/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Wk 7 - Naive Bayes/titanic-NB-pred.csv", row.names=FALSE)

#For more information about naive Bayes in e1071, see the manual at https://cran.r-project.org/web/packages/e1071/e1071.pdf

#Lesson 7.11 R Questions 
#What preprocessing steps did you do on the data?
- For preprocessing, I discretized "survived" and "PClass" and converted it from ordinal to nominal. I then removed the variables that I wouldn't be using such as "embarked" and saved this new dataset as "newtest" and "newtrain". Lastly, I handled the missing "na"'s from the dataset. 

#What algorithm parameters did you change? If using default parameters, say "default."
- In order to "smooth" the categorical data, I had to place the parameter of "smoothing". This is also in order to solve the handle the issue of zero probability.

#What evaluation method did you use? For example, if using CV, specify the number of fold.

#Evaluate CV accuracy on the training data