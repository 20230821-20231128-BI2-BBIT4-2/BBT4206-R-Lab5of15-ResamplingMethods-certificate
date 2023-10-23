# STEP 1. Install and Load the Required Packages ----
## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

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



### The Pima Indians Diabetes Dataset ----
# Execute the following to load the "Pima Indians Diabetes" dataset from the
# mlbench package:
data("PimaIndiansDiabetes")


# DATASET 1 (Splitting the dataset): Dow Jones Index ---
summary(PimaIndiansDiabetes)

#stracture of the dataset
str(PimaIndiansDiabetes)

## 1. Split the dataset ====
# Define a 75:25 train:test data split of the dataset.
# That is, 75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.75,
                                   list = FALSE)
PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]


## 2. Train a Naive Bayes classifier using the training dataset ----

PimaIndiansDiabetest_model_nb <-
  e1071::naiveBayes(diabetes ~ .,
                    data = PimaIndiansDiabetes_train)

## 3. Test the trained model using the testing dataset ----
### 3.a. Test the trained e1071 Naive Bayes model using the testing dataset ----
predictions_nb_e1071 <-
  predict(PimaIndiansDiabetest_model_nb,
          PimaIndiansDiabetes_test[, c("pregnant","glucose","pressure","triceps","mass","pedigree","age","insulin")])

## 4. View the Results ----
### 4.a. e1071 Naive Bayes model and test results using a confusion matrix ----

print(predictions_nb_e1071)

plot(table(predictions_nb_e1071,
                       PimaIndiansDiabetes_test[, c("pregnant","glucose","pressure","triceps","mass","pedigree","age","insulin","diabetes")]$diabetes))



## 5. Classification: SVM with Repeated k-fold Cross Validation ----

train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

PimaIndiansDiabetes_model_svm <-
  caret::train(`diabetes` ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "svmLinearWeights2", metric = "Accuracy")

### 5.b. Test the trained SVM model using the testing dataset ----
predictions_svm <- predict(PimaIndiansDiabetes_model_svm, PimaIndiansDiabetes_test[, 1:8])

### 5.c. View a summary of the model and view the confusion matrix ----
print(PimaIndiansDiabetes_model_svm)
caret::confusionMatrix(predictions_svm, PimaIndiansDiabetes_test$diabetes)
