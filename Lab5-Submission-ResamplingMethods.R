# STEP 1. Install and Load the Required Packages ----
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# DATASET 1 (Splitting the dataset): Dow Jones Index ----
student_performance <- read_csv(
  "data/StudentPerformanceDataset.csv",
  col_types = cols(
    gender = col_factor(levels = c("1", "0")),
    regret_choosing_bi = col_factor(levels = c("1", "0")),
    motivator = col_factor(levels = c("1", "0")),
    paid_tuition = col_factor(levels = c("1", "0")),
    extra_curricular = col_factor(levels = c("1", "0")),
    read_content_before_lecture = col_factor(levels = c("1", "2", "3", "4", "5")),
    anticipate_test_questions = col_factor(levels = c("1", "2", "3", "4", "5")),
    health = col_factor(levels = c("1", "2", "3", "4", "5")),
    GRADE = col_factor(levels = c("A", "B", "C", "D", "E"))
  )
)
summary(student_performance)

# The str() function is used to compactly display the structure (variables
# and data types) of the dataset
str(student_performance)

## 1. Split the dataset ====
# Define a 75:25 train:test data split of the dataset.
# That is, 75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(stock_ror_dataset$stock,
                                   p = 0.75,
                                   list = FALSE)
stock_ror_dataset_train <- stock_ror_dataset[train_index, ]
stock_ror_dataset_test <- stock_ror_dataset[-train_index, ]

## 2. Train a Naive Bayes classifier using the training dataset ----

### 2.a. OPTION 1: naiveBayes() function in the e1071 package ----
# The "naiveBayes()" function (case sensitive) in the "e1071" package
# is less sensitive to missing values hence all the features (variables
# /attributes) are considered as independent variables that have an effect on
# the dependent variable (stock).

stock_ror_dataset_model_nb_e1071 <- # nolint
  e1071::naiveBayes(stock ~ quarter + date + open + high + low + close +
                      volume + percent_change_price +
                      percent_change_volume_over_last_wk +
                      previous_weeks_volume + next_weeks_open +
                      next_weeks_close + percent_change_next_weeks_price +
                      days_to_next_dividend + percent_return_next_dividend,
                    data = stock_ror_dataset_train)

# The above code can also be written as follows to show a case where all the
# variables are being considered (stock ~ .):
stock_ror_dataset_model_nb <-
  e1071::naiveBayes(stock ~ .,
                    data = stock_ror_dataset_train)

### 2.b. OPTION 2: naiveBayes() function in the caret package ====
# The second option uses the caret::train() function in the caret package to
# train a Naive Bayes classifier but without the attributes that have missing
# values.
stock_ror_dataset_model_nb_caret <- # nolint
  caret::train(stock ~ ., data =
               stock_ror_dataset_train[, c("quarter", "date", "open",
                                           "high", "low", "close",
                                           "volume",
                                           "percent_change_price",
                                           "next_weeks_open",
                                           "next_weeks_close",
                                           "percent_change_next_weeks_price",
                                           "days_to_next_dividend",
                                           "percent_return_next_dividend",
                                           "stock")],
               method = "naive_bayes")

## 3. Test the trained model using the testing dataset ----
### 3.a. Test the trained e1071 Naive Bayes model using the testing dataset ----
predictions_nb_e1071 <-
  predict(stock_ror_dataset_model_nb_e1071,
          stock_ror_dataset_test[, c("quarter", "date", "open", "high",
                                     "low", "close", "volume",
                                     "percent_change_price",
                                     "percent_change_volume_over_last_wk",
                                     "previous_weeks_volume", "next_weeks_open",
                                     "next_weeks_close",
                                     "percent_change_next_weeks_price",
                                     "days_to_next_dividend",
                                     "percent_return_next_dividend")])

### 3.b. Test the trained caret Naive Bayes model using the testing dataset ----
predictions_nb_caret <-
  predict(stock_ror_dataset_model_nb_caret,
          stock_ror_dataset_test[, c("quarter", "date", "open", "high",
                                     "low", "close", "volume",
                                     "percent_change_price", "next_weeks_open",
                                     "next_weeks_close",
                                     "percent_change_next_weeks_price",
                                     "days_to_next_dividend",
                                     "percent_return_next_dividend")])

## 4. View the Results ----
### 4.a. e1071 Naive Bayes model and test results using a confusion matrix ----
# Please watch the following video first: https://youtu.be/Kdsp6soqA7o
print(predictions_nb_e1071)
caret::confusionMatrix(predictions_nb_e1071,
                       stock_ror_dataset_test[, c("quarter", "date", "open",
                                                  "high", "low", "close",
                                                  "volume",
                                                  "percent_change_price",
                                                  "percent_change_volume_over_last_wk", # nolint
                                                  "previous_weeks_volume",
                                                  "next_weeks_open",
                                                  "next_weeks_close",
                                                  "percent_change_next_weeks_price", # nolint
                                                  "days_to_next_dividend",
                                                  "percent_return_next_dividend", # nolint
                                                  "stock")]$stock)
plot(table(predictions_nb_e1071,
           stock_ror_dataset_test[, c("quarter", "date", "open", "high", "low",
                                      "close", "volume", "percent_change_price",
                                      "percent_change_volume_over_last_wk",
                                      "previous_weeks_volume",
                                      "next_weeks_open", "next_weeks_close",
                                      "percent_change_next_weeks_price",
                                      "days_to_next_dividend",
                                      "percent_return_next_dividend",
                                      "stock")]$stock))

### 4.b. caret Naive Bayes model and test results using a confusion matrix ----
print(stock_ror_dataset_model_nb_caret)
caret::confusionMatrix(predictions_nb_caret,
                       stock_ror_dataset_test[, c("quarter", "date", "open",
                                                  "high", "low", "close",
                                                  "volume",
                                                  "percent_change_price",
                                                  "percent_change_volume_over_last_wk", # nolint
                                                  "previous_weeks_volume",
                                                  "next_weeks_open",
                                                  "next_weeks_close",
                                                  "percent_change_next_weeks_price", # nolint
                                                  "days_to_next_dividend",
                                                  "percent_return_next_dividend", # nolint
                                                  "stock")]$stock)
plot(table(predictions_nb_caret,
           stock_ror_dataset_test[, c("quarter", "date", "open", "high", "low",
                                      "close", "volume", "percent_change_price",
                                      "percent_change_volume_over_last_wk",
                                      "previous_weeks_volume",
                                      "next_weeks_open", "next_weeks_close",
                                      "percent_change_next_weeks_price",
                                      "days_to_next_dividend",
                                      "percent_return_next_dividend",
                                      "stock")]$stock))

# DATASET 2 (Splitting the dataset): Default of credit card clients ----
defaulter_dataset <-
  readr::read_csv(
    "data/default of credit card clients.csv",
    col_types = cols(
      SEX = col_factor(levels = c("1", "2")),
      EDUCATION = col_factor(levels = c("0", "1", "2", "3", "4", "5", "6")),
      MARRIAGE = col_factor(levels = c("0", "1", "2", "3")),
      `default payment next month` = col_factor(levels = c("1", "0")),
      `default payment next month` = col_factor(levels = c("1", "0"))
    ),
    skip = 1
  )
summary(defaulter_dataset)
str(defaulter_dataset)

## 1. Split the dataset ----
# Define an 80:20 train:test split ratio of the dataset
# (80% of the original data will be used to train the model and 20% of the
# original data will be used to test the model).
train_index <- createDataPartition(defaulter_dataset$`default payment next month`, # nolint
                                   p = 0.80, list = FALSE)
defaulter_dataset_train <- defaulter_dataset[train_index, ]
defaulter_dataset_test <- defaulter_dataset[-train_index, ]

## 2. Train a Naive Bayes classifier using the training dataset ----

### 2.a. OPTION 1: "NaiveBayes()" function in the "klaR" package ----
defaulter_dataset_model_nb_klaR <- # nolint
  klaR::NaiveBayes(`default payment next month` ~ .,
                   data = defaulter_dataset_train)

### 2.b. OPTION 2: "naiveBayes()" function in the e1071 package ----
defaulter_dataset_model_nb_e1071 <- # nolint
  e1071::naiveBayes(`default payment next month` ~ .,
                    data = defaulter_dataset_train)

## 3. Test the trained Naive Bayes model using the testing dataset ----
predictions_nb_e1071 <-
  predict(defaulter_dataset_model_nb_e1071,
          defaulter_dataset_test[, 1:25])

## 4. View the Results ----
### 4.a. e1071 Naive Bayes model and test results using a confusion matrix ----
print(defaulter_dataset_model_nb_e1071)
caret::confusionMatrix(predictions_nb_e1071,
                       defaulter_dataset_test$`default payment next month`)
# The confusion matrix can also be viewed graphically,
# although with less information.
plot(table(predictions_nb_e1071,
           defaulter_dataset_test$`default payment next month`))

# DATASET 3 (Bootstrapping): Daily Demand Forecasting Orders Data Set =====
demand_forecasting_dataset <-
  readr::read_delim(
    "data/Daily_Demand_Forecasting_Orders.csv",
    delim = ";",
    escape_double = FALSE,
    col_types = cols(
      `Week of the month (first week, second, third, fourth or fifth week` =
        col_factor(levels = c("1", "2", "3", "4", "5")),
      `Day of the week (Monday to Friday)` =
        col_factor(levels = c("2", "3", "4", "5", "6"))
    ),
    trim_ws = TRUE
  )
summary(demand_forecasting_dataset)
str(demand_forecasting_dataset)

## 1. Split the dataset ----
demand_forecasting_dataset_cor <- cor(demand_forecasting_dataset[, 3:13])
View(demand_forecasting_dataset_cor)
# Define a 75:25 train:test data split ratio of the dataset
# (75% of the original data will be used to train the model and 25% of the
# original data will be used to test the model)
train_index <-
  createDataPartition(demand_forecasting_dataset$`Target (Total orders)`,
                      p = 0.75, list = FALSE)
demand_forecasting_dataset_train <- demand_forecasting_dataset[train_index, ] # nolint
demand_forecasting_dataset_test <- demand_forecasting_dataset[-train_index, ] # nolint

## 2. Train a linear regression model (for regression) ----

### 2.a. Bootstrapping train control ----
# The "train control" allows you to specify that bootstrapping (sampling with
# replacement) can be used and also the number of times (repetitions or reps)
# the sampling with replacement should be done. The code below specifies
# bootstrapping with 500 reps. (common values for reps are thousands or tens of
# thousands depending on the hardware resources available).

# This increases the size of the training dataset from 48 observations to
# approximately 48 x 500 = 24,000 observations for training the model.
train_control <- trainControl(method = "boot", number = 500)

demand_forecasting_dataset_model_lm <- # nolint
  caret::train(`Target (Total orders)` ~
                 `Non-urgent order` + `Urgent order` +
                   `Order type A` + `Order type B` +
                   `Order type C` + `Fiscal sector orders` +
                   `Orders from the traffic controller sector` +
                   `Banking orders (1)` + `Banking orders (2)` +
                   `Banking orders (3)`,
               data = demand_forecasting_dataset_train,
               trControl = train_control,
               na.action = na.omit, method = "lm", metric = "RMSE")

## 3. Test the trained linear regression model using the testing dataset ----
predictions_lm <- predict(demand_forecasting_dataset_model_lm,
                          demand_forecasting_dataset_test[, 1:13])

## 4. View the RMSE and the predicted values for the 12 observations ----
print(demand_forecasting_dataset_model_lm)
print(predictions_lm)

## 5. Use the model to make a prediction on unseen new data ----
# New data for each of the 12 variables (independent variables) that determine
# the dependent variable can also be specified as follows in a data frame:
new_data <-
  data.frame(`Week of the month (first week, second, third, fourth or fifth week` = c(1), # nolint
             `Day of the week (Monday to Friday)` = c(2),
             `Non-urgent order` = c(151.06),
             `Urgent order` = c(132.11), `Order type A` = c(52.11),
             `Order type B` = c(109.23),
             `Order type C` = c(160.11), `Fiscal sector orders` = c(7.832),
             `Orders from the traffic controller sector` = c(52112),
             `Banking orders (1)` = c(20130), `Banking orders (2)` = c(94788),
             `Banking orders (3)` = c(12610), check.names = FALSE)

# The variables that are factors (categorical) in the training dataset must
# also be defined as factors in the new data
new_data$`Week of the month (first week, second, third, fourth or fifth week` <-
  as.factor(new_data$`Week of the month (first week, second, third, fourth or fifth week`) # nolint

new_data$`Day of the week (Monday to Friday)` <-
  as.factor(new_data$`Day of the week (Monday to Friday)`)

# We now use the model to predict the output based on the unseen new data:
predictions_lm_new_data <-
  predict(demand_forecasting_dataset_model_lm, new_data)

# The output below refers to the total orders:
print(predictions_lm_new_data)

# DATASET 4 (CV, Repeated CV, and LOOCV): Iranian Churn Dataset ----
churn_dateset <- read_csv(
  "data/Customer Churn.csv",
  col_types = cols(
    Complains = col_factor(levels = c("0",
                                      "1")),
    `Age Group` = col_factor(levels = c("1",
                                        "2", "3", "4", "5")),
    `Tariff Plan` = col_factor(levels = c("1",
                                          "2")),
    Status = col_factor(levels = c("1",
                                   "2")),
    Churn = col_factor(levels = c("0",
                                  "1"))
  )
)
summary(churn_dateset)
str(churn_dateset)

## 1. Split the dataset ====
# define a 75:25 train:test split of the dataset
train_index <- createDataPartition(churn_dateset$`Customer Value`,
                                   p = 0.75, list = FALSE)
churn_dateset_train <- churn_dateset[train_index, ]
churn_dateset_test <- churn_dateset[-train_index, ]

## 2. Regression: Linear Model ----
### 2.a. 10-fold cross validation ----

# Please watch the following video first: https://youtu.be/fSytzGwwBVw
# The train control allows you to specify that k-fold cross validation
# can be used as well as the number of folds (common folds are 5-fold and
# 10-fold cross validation).

# The k-fold cross-validation method involves splitting the dataset (training
# dataset) into k-subsets. Each subset is held-out (withheld) while the model is
# trained on all other subsets. This process is repeated until the accuracy/RMSE
# is determined for each instance in the dataset, and an overall accuracy/RMSE
# estimate is provided.

train_control <- trainControl(method = "cv", number = 10)

churn_dateset_model_lm <-
  caret::train(`Customer Value` ~ .,
               data = churn_dateset_train,
               trControl = train_control, na.action = na.omit,
               method = "lm", metric = "RMSE")

### 2.b. Test the trained linear model using the testing dataset ----
predictions_lm <- predict(churn_dateset_model_lm, churn_dateset_test[, -13])

### 2.c. View the RMSE and the predicted values ====
print(churn_dateset_model_lm)
print(predictions_lm)

## 3. Classification: LDA with k-fold Cross Validation ----

### 3.a. LDA classifier based on a 5-fold cross validation ----
# We train a Linear Discriminant Analysis (LDA) classifier based on a 5-fold
# cross validation train control but this time, using the churn variable for
# classification, not the customer value variable for regression.
train_control <- trainControl(method = "cv", number = 5)

churn_dateset_model_lda <-
  caret::train(`Churn` ~ ., data = churn_dateset_train,
               trControl = train_control, na.action = na.omit, method = "lda2",
               metric = "Accuracy")

### 3.b. Test the trained LDA model using the testing dataset ----
predictions_lda <- predict(churn_dateset_model_lda,
                           churn_dateset_test[, 1:13])

### 3.c. View the summary of the model and view the confusion matrix ----
print(churn_dateset_model_lda)
caret::confusionMatrix(predictions_lda, churn_dateset_test$Churn)

## 4. Classification: Naive Bayes with Repeated k-fold Cross Validation ----
### 4.a. Train an e1071::naive Bayes classifier based on the churn variable ----
churn_dateset_model_nb <-
  e1071::naiveBayes(`Churn` ~ ., data = churn_dateset_train)

### 4.b. Test the trained naive Bayes classifier using the testing dataset ----
predictions_nb_e1071 <-
  predict(churn_dateset_model_nb, churn_dateset_test[, 1:14])

### 4.c. View a summary of the naive Bayes model and the confusion matrix ----
print(churn_dateset_model_nb)
caret::confusionMatrix(predictions_nb_e1071, churn_dateset_test$Churn)

## 5. Classification: SVM with Repeated k-fold Cross Validation ----
### 5.a. SVM Classifier using 5-fold cross validation with 3 reps ----
# We train a Support Vector Machine (for classification) using "Churn" variable
# in the training dataset based on a repeated 5-fold cross validation train
# control with 3 reps.

# The repeated k-fold cross-validation method involves repeating the number of
# times the dataset is split into k-subsets. The final model accuracy/RMSE is
# taken as the mean from the number of repeats.

train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

churn_dateset_model_svm <-
  caret::train(`Churn` ~ ., data = churn_dateset_train,
               trControl = train_control, na.action = na.omit,
               method = "svmLinearWeights2", metric = "Accuracy")

### 5.b. Test the trained SVM model using the testing dataset ----
predictions_svm <- predict(churn_dateset_model_svm, churn_dateset_test[, 1:13])

### 5.c. View a summary of the model and view the confusion matrix ----
print(churn_dateset_model_svm)
caret::confusionMatrix(predictions_svm, churn_dateset_test$Churn)

## 6. Classification: Naive Bayes with Leave One Out Cross Validation ----
# In Leave One Out Cross-Validation (LOOCV), a data instance is left out and a
# model constructed on all other data instances in the training set. This is
# repeated for all data instances.

### 6.a. Train a Naive Bayes classifier based on an LOOCV ----
train_control <- trainControl(method = "LOOCV")

churn_dateset_model_nb_loocv <-
  caret::train(`Churn` ~ ., data = churn_dateset_train,
               trControl = train_control, na.action = na.omit,
               method = "naive_bayes", metric = "Accuracy")

### 6.b. Test the trained model using the testing dataset ====
predictions_nb_loocv <-
  predict(churn_dateset_model_nb_loocv, churn_dateset_test[, 1:14])

### 6.c. View the confusion matrix ====
print(churn_dateset_model_nb_loocv)
caret::confusionMatrix(predictions_nb_loocv, churn_dateset_test$Churn)
