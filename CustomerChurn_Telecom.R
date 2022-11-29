library(caret)
library(ggplot2)
library(lattice)
library(pROC)
library(forecast)
library(caTools)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ipred)
library(Metrics)

#Loading the file.
customer_df <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", stringsAsFactors = TRUE)

#Drop columns. 
#Remove customer ID, Gender.
customer_df <- customer_df[,-1:-2]

#Convert Contract to Numeric
customer_df$Contract <- as.numeric(customer_df$Contract)
customer_df$Contract[customer_df$Contract==1] <- 1
customer_df$Contract[customer_df$Contract==2] <- 12
customer_df$Contract[customer_df$Contract==3] <- 24

#Recoding the Variable Churn, Partner, Dependents, PhoneService, Paperless Billing to 0s 1s. 
customer_df$Churn <- as.numeric(customer_df$Churn == "Yes")
customer_df$Partner <- as.numeric(customer_df$Partner == "Yes")
customer_df$Dependents <- as.numeric(customer_df$Dependents == "Yes")
customer_df$PhoneService <- as.numeric(customer_df$PhoneService == "Yes")
customer_df$PaperlessBilling <- as.numeric(customer_df$PaperlessBilling == "Yes")
str(customer_df)

#Senior Citizen, tenure to numerics. 
customer_df$SeniorCitizen <- as.numeric(customer_df$SeniorCitizen)
customer_df$tenure <- as.numeric(customer_df$tenure)
str(customer_df)

#Deduce No internet service as No & No Phone service as No. 
customer_df <- data.frame(lapply(customer_df, function(x) {
  gsub("No internet service", "No", x)}))

customer_df <- data.frame(lapply(customer_df, function(x) {
  gsub("No phone service", "No", x)}))

str(customer_df)
#Ensuring that Numerical Variables are Numeric. 
num_columns <- c("Contract", "tenure", "MonthlyCharges", "TotalCharges")
customer_df[num_columns] <- sapply(customer_df[num_columns], as.numeric)

# Convert all char columns to factor
customer_df <- as.data.frame(unclass(customer_df),stringsAsFactors = TRUE)
str(customer_df)

#We can now either impute the data for Total Charges Feature by calculating mean or dropping the rows. 
#Approach One: Calculating the mean for the TotalCharges Column.
# compute each column's mean using mean() function
mean_value <- mean(customer_df$TotalCharges,na.rm = TRUE)

#Replacing missing values in the totalCharges column with the calculated mean. 
customer_df$TotalCharges[is.na(customer_df$TotalCharges)] <- mean_value
str(customer_df)

#Inspecting the Data for missing values:
sum(is.na(customer_df))

#Data Partition:
set.seed(123)
train.index <- sample(1:nrow(customer_df), nrow(customer_df) * 0.7)
train.df <- customer_df[train.index,]
valid.df <- customer_df[-train.index,]

#Tree Classification:

#Select Predictors which are important for classification.
#Training
default.ct <- rpart(Churn~Contract+tenure+TotalCharges, data = train.df, method = "class")
rpart.plot(default.ct, extra = 1)

#Feature Importance
default.ct$variable.importance

#Prediction
default.ct.point.pred <- predict(default.ct, valid.df, type = "class")

#Confusion Matrix for Classification Tree
confusionMatrix(default.ct.point.pred, valid.df$Churn)

#Logisitic Regression:
logit.reg <- glm(Churn ~., data = train.df, family = "binomial") 
summary(logit.reg)

# use predict() with type = "response" to compute predicted probabilities
# if type not specified, log-odds will be returned
logit.reg.pred <- predict(logit.reg, valid.df,  type = "response")

#Plot RoC curve. 
r <- roc(valid.df$Churn, logit.reg.pred)
plot.roc(r)

# find the best threshold
coord_vec <- coords(r, x = "best")

# Choose cutoff value and evaluate classification performance
pred <- ifelse(logit.reg.pred > 0.2721008, 1, 0)

# generate the confusion matrix based on the prediction
confusionMatrix(factor(pred), factor(valid.df$Churn), positive = "1")

#Bagged Trees
model <- bagging(Churn~., data = train.df, coob = TRUE)
pred <- predict(model,newdata = valid.df, type = "prob")
auc(actual = valid.df$Churn, predicted = pred[,"1"]) 

#Random Forest Model:
customer_model.rf <- randomForest(Churn ~ ., data=train.df)

#Predicting on the validation set and checking the Confusion Matrix.
testPred <- predict(customer_model.rf, newdata=valid.df)

#Confusion Matrix
confusionMatrix(valid.df$Churn, testPred)

#Checking the variable Importance Plot - Gini Index. 
varImpPlot(customer_model.rf)
