# Practical Machine Learning - Project Writeup



## Synopsis
In this project we were given data on quantified self movement and asked to build a machine learning algorithm. Data was recorded from accelerometers on the belt, forearm, arm & dumbell of six participants. These participants were asked to perform certain excercises with weights correctly and incorrectly in five different ways. Each way was recorded as either classe A, B, C, D or E. The goal of this project was to use this data to build a machine learning algorithm from the accleromiter data that can accurately predict which way the exercise was performed, classe A, B, C, D or E. The data came from this study - http://groupware.les.inf.puc-rio.br/har and the raw training and test data can be found at the below links:

* Training Data - https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
* Test Data - https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


## Data Processing & Model Selection
To start this project I loaded both the training and test data into R. Looking at the training data I immediately noticed there were many columns that were almost entirely empty or NA, I then looked at the test data and discovered it also had many columns with no data. My first step to clean up the training data was to exclude any of the columns with no data as they won't be useful in building a prediction algorithm. To clean up the training data a bit more I went through and excluded any columns that weren't measurements from the activity monitors, as those also won't be used in the algorithm. 

```r
dat <- read.csv("pml-training.csv") # read in training data file
test <- read.csv("pml-testing.csv") # read in testing data file
test <- test[,colSums(is.na(test))==0] # subset actual test data to exclude NA columns

# column names to keep in training set, excludes columns that aren't activity monitors
colKeep <- names(subset(test, select=-c(X, user_name, cvtd_timestamp, raw_timestamp_part_1, 
                                        raw_timestamp_part_2, new_window, num_window, problem_id))) 
colKeep <- c(colKeep, "classe") # add 'classe' colunm to list of columns to keep

datSub <- dat[, colKeep] # subset training data to exclude unnecessary columns
```

Now I'm ready to start the model selection process, I need to find what are the most important variables to use when building the model. First I used set.seed for reproducibilty, and because the dataset is large I segmented 10% of the data to use for my model selection, then I further segmented that data into 70% for training and 30% for testing. I chose to use the Random Forests method for my model because despite being slow to calculate they are quite accurate. So I used the train function from the caret package to build a test model for the classe outcome using all other variables as predictors.

```r
library(caret)
set.seed(55684) # for reproducibility
inTrain1 <- createDataPartition(y=datSub$classe, p=.1, list=FALSE) 
training1 <- datSub[inTrain1, ] # segment 10% of training data for model selection process

inTrain2 <- createDataPartition(y=training1$classe, p=.7, list=FALSE)

training2 <- training1[inTrain2, ] # create 70% of segmented data for training
testing2 <- training1[-inTrain2, ] # create 30% of segmented data for testing

# using Random Forests method, create a test model using 'classe' as outcome and all remaining columns as predictors
testFit <- train(classe ~ ., data=training2, method="rf", prox=TRUE)
```


```r
testFit
```

```
## Random Forest 
## 
## 1377 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 1377, 1377, 1377, 1377, 1377, 1377, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    0.9       0.9    0.01         0.02    
##   27    0.9       0.9    0.01         0.02    
##   52    0.9       0.9    0.02         0.02    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
pred <- predict(testFit, testing2) # use testFit to predict on our segmented testing data
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
testing2$predRight <- pred==testing2$classe
testAcc <- confusionMatrix(table(pred, testing2$classe))$overall[1] # accuracy calculation
table(pred, testing2$classe) # make a table to show accuracy of testFit on segmented test data
```

```
##     
## pred   A   B   C   D   E
##    A 165  10   0   1   0
##    B   0 101   2   2   1
##    C   0   3  97   6   0
##    D   2   0   3  85   3
##    E   0   0   0   2 104
```
You can see from the results above the model is fairly accurate at ```testAcc``` = 0.9404, but still contains way too many predictors. To trim down the 50 something predictors I used in my testFit model I use the function 'varImp' from the caret package. I pass my testFit model to this funtion and it gives me the top 20 variables in order of importance, you can see the output from this below. These will be the variables I include in my final model.


```r
varImp(testFit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                   Overall
## roll_belt           100.0
## pitch_forearm        78.2
## magnet_dumbbell_z    62.8
## yaw_belt             53.2
## pitch_belt           41.8
## magnet_dumbbell_y    41.3
## roll_forearm         36.2
## roll_dumbbell        28.6
## accel_forearm_x      23.3
## accel_dumbbell_y     22.9
## magnet_dumbbell_x    21.0
## accel_belt_z         20.7
## magnet_belt_y        18.3
## magnet_belt_z        18.0
## accel_dumbbell_z     16.1
## gyros_belt_z         15.7
## accel_dumbbell_x     13.9
## magnet_forearm_z     12.8
## yaw_dumbbell         11.8
## gyros_dumbbell_y     11.6
```

## Final Model & Cross Validation
So now I have the list of the 20 most important variables to include in my final model, so I make a subset of the training data to include only the columns necessary. I again use set.seed for reproducibility and then segment the training data into 70% to use in my model and 30% for testing/cross validation. This will also allow me to obtain an Out of Sample Error estimate. So I then make my final model, again using the Random Forests method with my training data.

```r
set.seed(55684) # for reproducibility

# subset pml-training.csv data to only include most important predictors accroding to varImp
datFinal <- subset(dat, select=c(roll_belt, pitch_forearm, magnet_dumbbell_z, 
        yaw_belt, pitch_belt, magnet_dumbbell_y, roll_forearm, roll_dumbbell, 
        accel_forearm_x, accel_dumbbell_y, magnet_dumbbell_x, accel_belt_z, 
	magnet_belt_y, magnet_belt_z, accel_dumbbell_z, gyros_belt_z, accel_dumbbell_x, 
	magnet_forearm_z, yaw_dumbbell, gyros_dumbbell_y, classe))

inTrain <- createDataPartition(y=datFinal$classe, p=.7, list=FALSE)
training <- datFinal[inTrain, ] # create 70% of final data for training
testing <- datFinal[-inTrain, ] # create 30% of final data for testing

# again use Random Forests method to create final model using 'classe' as outcome and most important variables as predictors
finalFit <- train(classe ~ ., data=training, method="rf", prox=TRUE)
```

You can see below that between the output of finalFit and the accuracy table (compares the predictions from my final model vs the actual data), that my final model is extremely accurate:

```r
finalFit
```

```
## Random Forest 
## 
## 13737 samples
##    20 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    1         1      0.002        0.003   
##   11    1         1      0.003        0.003   
##   20    1         1      0.004        0.005   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 11.
```


```r
finalPred <- predict(finalFit, testing) # use finalFit to predict on testing data
testing$predRight <- finalPred==testing$classe
finalAcc <- confusionMatrix(table(finalPred, testing$classe))$overall[1] # accuracy calculation
table(finalPred, testing$classe) # make a table to show accuracy of finalFit on testing data
```

```
##          
## finalPred    A    B    C    D    E
##         A 1668    8    1    0    0
##         B    2 1122    1    1    1
##         C    3    7 1016   12    6
##         D    0    2    8  950    1
##         E    1    0    0    1 1074
```

## Conclusion & Predictions
Using the predict function with my final model on my testing data I cross validate for accuracy and obtain an out of sample error rate. The accuracy of my final model on the testing data is calculated in the ```finalAcc``` variable and = 0.9907, meaning we get an out of sample error rate of 0.0093. I am quite pleased with the accuracy of the final model and am confident it will successfully predict the classe for each of the submissions in the test data for this project. 


```r
predictions <- predict(finalFit, newdata=test)
```
So my final predictions for the test data are - ```predictions``` = B, A, B, A, A, E, D, B, A, A, B, C, B, A, E, E, A, B, B, B
