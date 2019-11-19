
'''
Title: "Homework 6 - Naive Bayes & Decision Tree for Handwriting Recognition"
Student: "Funmi Esuruoso"
date: "November 19, 2019'''

'''Section 1: Introduction - Preprocess (Import data and load packages)
Briefly describe the classification problem and general data preprocessing. Note that 
some data preprocessing steps maybe specific to a particular algorithm. Report those 
steps under each algorithm section.'''
library(dplyr)
library(caret)
library(rpart)
library(e1071)
library(stringr)

#from mac
digittrain <- read.csv("/Volumes/STORE N GO/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Deliverables/HWs/HW6/digittrain.csv")
digittest <- read.csv("/Volumes/STORE N GO/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Deliverables/HWs/HW6/digittest.csv")

#from windows
digittrain <- read.csv("F:/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Deliverables/HWs/HW6/digittrain.csv")
digittest <- read.csv("F:/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Deliverables/HWs/HW6/digittest.csv")


dim(digittrain)
# We observe that the training dataset has 42,000 observations and 785 variables. 
str(digittrain[, 1:10])
summary(digittrain[, 1:10])

dim(digittest)
# We observe that the test dataset has 28,000 observations and 784 variables. 
str(digittest[, 1:10])
summary(digittest[, 1:10])

# The dataset is 0 through 9 handwritten images & 784 pixel variables. Our labels are "Pixels" and our "data type are all integers 

digittrain$label <- as.factor(digittrain$label)  #we don't really need this OMIT???

'''
We will be working with a sample of the data (due to the size) prior to building our model. 
As described in the "Task Description", we will be sampling 10% of our training and testing
data to train and test to be used for our classifier.'''

#we set the seed for randomness
set.seed(150)

# selection of 10% of the training & test data
splitrain <- sample(nrow(digittrain), nrow(digittrain) * .1)
splitest <- sample(nrow(digittest), nrow(digittest) * .1)

# row selection to build our set 
subtrain <- digittrain[splitrain, ]
subtest <- digittest[splitest, ]

#specify that the row names are "NULL", that there are none
row.names(digittrain) <- NULL
row.names(digittest) <- NULL

'''Aside from that, we are going to try to use the pixel information - which pixel lights 
up - to determine which number is formed, and use that to try and determine the correct 
labels for the test dataset.''' 



'''Section No. 2: Decision Tree
Build a decision tree model. Tune the parameters, such as the pruning options, and report 
the 3-fold CV accuracy.'''

# Decision tree model
digittreetrain <- rpart(label ~ ., data = subtrain, method = 'class', 
                      control = rpart.control(cp = 0), minsplit = 100, maxdepth = 10)

# Testing the accuracy of the tree on the training set
trainacc <- data.frame(predict(digittreetrain, subtrain))

#choose max likelihood
trainacc <- as.data.frame(names(trainacc[apply(trainacc, 1, which.max)]))
colnames(trainacc) <- 'prediction'
trainacc$number <- substr(trainacc$prediction, 2, 2)
trainacc <- subtrain %>% bind_cols(trainacc) %>% select(label, number) %>% mutate(label = as.factor(label), number = as.factor(round(as.numeric(number), 0)))

# Now let's build the Confusion matrix so we can examine the accuracy percentage
confusionMatrix(trainacc$label, trainacc$number)
#our training set gives us an 86% (approximate) accuracy.

# now we run the same prediction on the test data
testacc <- data.frame(predict(digittreetrain, subtest))
testacc <- as.data.frame(names(testacc[apply(testacc, 1, which.max)]))
colnames(testacc) <- 'prediction'
testacc$number <- substr(testacc$prediction, 2, 2)

# Seems about even but we cannot test the results unless we run the decision tree on the entire test set and sumbmit it to Kaggle. 

finaltest <- data.frame(predict(digittreetrain, digittest))
finaltest <- as.data.frame(names(finaltest[apply(finaltest, 1, which.max)]))
colnames(finaltest) <- 'ImageId'
finaltest$Label <- substr(finaltest$ImageId, 2, 2)
finaltest$ImageId <- 1:nrow(finaltest)

#Now we can export the model (file) and view Kaggle results
write.csv(finaltest, file="/Volumes/STORE N GO/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Deliverables/HWs/HW6/FEsuruoso - Digit Decision Tree Test.csv")

''' After submitting my predictions to Kaggle for the competition, I scored 0.75. Its not terrible but
could have been better. In my next submission, I would want to use a larger percentage of my data
and also porform additional and more extensive tuning.'''  

'''Section 3: Naive bayes
Now that we have tried identifying the numbers using a decision tree, 
it is time to try our hand using the Naive Bayes algorithm. '''

# first we build the classifier. 
digittrainnb <- naiveBayes(as.factor(label) ~ ., data = subtrain)

# Again, test the results on the train set before submitting to Kaggle. 
nbtrainacc <- predict(digittrainnb, subtrain, type = 'class')
confusionMatrix(nbtrainacc, as.factor(subtrain$label))
# The results aren't the best in our confusion matrix as we only have 57%  accuracy. 

nbtestacc <- predict(digittrainnb, digittest, type = 'class')
nbtestacc <- data.frame(nbtestacc)
colnames(nbtestacc)[1] <- 'Label'
nbtestacc$ImageId <- 1:nrow(nbtestacc)
nbtestacc <- nbtestacc %>% select(ImageId, Label)

#We will now submit our Naive Bayes algorithm to kaggle. First we need to export to csv.
#mac
write.csv(nbtestacc, file="/Volumes/STORE N GO/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Deliverables/HWs/HW6/FEsuruoso - Naive Bayes Classifier.csv", row.names = FALSE)
#windows
write.csv(nbtestacc, file="F:/00 - Graduate School/00 - SYR In Session/00 - Fall 2019/IST 707 - Data Analytics Wed9PM/Deliverables/HWs/HW6/FEsuruoso - Naive Bayes Classifier.csv", row.names = FALSE)

'''While there was no tuning done on the Naive Bayes classifier, I scored a 57% (approximate) on my Kaggle submission. This is not as good as our decision tree algorithm''' 


## Section 4: Algorithm performance comparison
We can see by the Confusion matrices shown above that the decision tree does a better job at classifying the results as it is missing less on many numbers. The naive Bayes classifier, however, has a lot of trouble distinguishing between the number 2 and the other numbers, meaning that the algorithm needs more work. 
But what has to be taken into account was that, due to available memory, only ten percent of the actual training set was used. This set was selected at random so, had we used another random set, or the entire training set, results might have been different. This is something worth investigating and pursuing, as it could lead to better results, given that the algorithms will have more data points from which to learn from. 

## Section 5: Kaggle test result
The Kaggle test results were not great but they were quite encouraging. Having the decision tree score 0.7607 was a great first step that can be improved upon. 
The naive Bayes algorithm, on the other hand, started off at 0.4984 - not a great result as it is almost a coin-toss between the actual result and the predicted result. The model needs further work, both in the data being used to build the classifier, and the tuning methods that can help with the model underfitting. 
