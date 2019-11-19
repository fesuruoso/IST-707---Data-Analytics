
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

#import datasets
digittrain <- read.csv("/Volumes/STORE N GO/00 - Graduate School/HW6/digittrain.csv")
digittest <- read.csv("/Volumes/STORE N GO/00 - Graduate School/HW6/digittest.csv")

dim(digittrain)

# We observe that the training dataset has 42,000 observations and 785 variables. 
str(digittrain[, 1:10])
summary(digittrain[, 1:10])

dim(digittest)
# We observe that the test dataset has 28,000 observations and 784 variables. 
str(digittest[, 1:10])
summary(digittest[, 1:10])

# The dataset is 0 through 9 handwritten images & 784 pixel variables. Our labels are "Pixels" and our "data type are all integers 

digittrain$label <- as.factor(digittrain$label)

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
write.csv(finaltest, file="/Volumes/STORE N GO/00 - Graduate School/HW6/FEsuruoso - Digit Decision Tree Test.csv")

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
write.csv(nbtestacc, file="/Volumes/STORE N GO/00 - Graduate School/HW6/FEsuruoso - Naive Bayes Classifier.csv", row.names = FALSE)
'''While there was no tuning done on the Naive Bayes classifier, I scored a 57% (approximate) on my Kaggle submission. This is not as good as our decision tree algorithm''' 

## Section 4: Algorithm performance comparison
I was not pleased with the results obtained by the Naïve Bayes algorithm. I received a 57% in my confidence matrix and a57% on my Kaggle submission.  I did not perform any pruning measures like I did with the previous model, so this might have a lot to do with it. There also weren’t any additional parameters set.  
One of the most significant measures of how good our models did was the Confusion Matrix accuracy. While we had around 80% for the first model, our second model only gave us around 57%. We see that the first model also properly identified our models better. It is also worthy to note that had we used a bigger portion of our data (we only used 10%) and tuned better, we might have had a better model. 

## Section 5: Kaggle test result
As shown above, I scored approximately 75% on my decision tree submission and 57% on my Naïve Bayes submission. Both scores could be improved greatly by the following:

•	Increasing the sample size in our models from 10% to a larger sample size (maybe 20% or 40%)
•	Additional tuning that addresses model fitting
•	Address specific model issues (i.e smaller dataset for Naïve Bayes and larger dataset for decision tree)
