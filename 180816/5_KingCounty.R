# Regression model predicting King County house prices

library(dummies) # dummy.data.frame
library(car) # scatterplotMatrix

## Set the working directory
setwd("C:/Users/user/Desktop/drive/kuai/2_Analysis/5_KSB/workspace")

## Multivariate linear regression
KC <- read.csv("KingCounty.csv")
str(KC)

# Pretreat the data
KC <- KC[,c(3:5, 12, 6, 7, 15, 9)]
str(KC)

# Check all NA
library(dplyr)
colSums(is.na(KC))


## Indices for the activated input variables and split the data
nObs <- dim(KC)[1]


## 1-C coding
dat <- dummy.data.frame(KC, names = c("waterfront"))
str(dat)


## Split the data into the training/validation sets
set.seed(2018040318)
trn_idx <- sample(1:nObs, round(0.7*nObs))
trn_data <- dat[trn_idx,]
val_data <- dat[-trn_idx,]
dim(trn_data)
dim(val_data)


## Train the MLR
full_model <- lm(price ~., data = trn_data)
summary(full_model)


## Plot the result for analyzing residuals
par(mfrow = c(1,3))

# residual plot
plot(fitted(full_model), residuals(full_model), 
     xlab = "Fitted values", ylab = "Residuals", main = "residual plot")
abline(0,0,lty = 3)

# box plot
boxplot(residuals(full_model), ylab = "Residuals", main = "box plot")

# normal probability plot
qqnorm(rstandard(full_model), xlab = "Normal scores", 
       ylab = "Standardized residuals", main = "Normal Q-Q")
abline(0,1,col = "red")

dev.off()


## Feature scaling for improve accuracy
scatterplotMatrix(~ price + bedrooms + bathrooms + condition + grade + 
                    sqft_living + sqft_lot + yr_built, 
                  data = trn_data, diagonal = "density", 
                  reg.line = FALSE, smoother = FALSE)
scatterplotMatrix(~ log(price) + bedrooms + bathrooms + condition + grade + 
                    sqft_living + sqft_lot + yr_built, 
                  data = trn_data, diagonal = "density", 
                  reg.line = FALSE, smoother = FALSE)

# Train the MLR
full_model_tfY <- lm(log(price) ~., data = trn_data)
summary(full_model_tfY)

## Plot the result for analyzing residuals
par(mfrow = c(1,3))

# residual plot
plot(fitted(full_model_tfY), residuals(full_model_tfY), 
     xlab = "Fitted values", ylab = "Residuals", main = "residual plot")
abline(0,0,lty = 3)

# box plot
boxplot(residuals(full_model_tfY), ylab = "Residuals", main = "box plot")

# normal probability plot
qqnorm(rstandard(full_model_tfY), xlab = "Normal scores",
       ylab = "Standardized residuals", main = "Normal Q-Q")
abline(0,1,col = "red")

dev.off()


## Compare all models by validation data
full_fit <- predict(full_model, newdata = val_data)
full_fit_tfY <- predict(full_model_tfY, newdata = val_data)
full_fit_tfY1 <- exp(full_fit_tfY)


## Performance matrix
# 1. Mean squared error (MSE)
perf_mat <- matrix(0, 4, 2)
perf_mat[1,1] <- mean((val_data$price - full_fit)^2)
perf_mat[1,2] <- mean((val_data$price - full_fit_tfY1)^2)

# 2. Root mean squared error (RMSE)
perf_mat[2,1] <- sqrt(mean((val_data$price - full_fit)^2))
perf_mat[2,2] <- sqrt(mean((val_data$price - full_fit_tfY1)^2))

# 3. Mean absolute error (MAE)
perf_mat[3,1] <- mean(abs(val_data$price - full_fit))
perf_mat[3,2] <- mean(abs(val_data$price - full_fit_tfY1))

# 4. Mean absolute percentage error (MAPE)
perf_mat[4,1] <- mean((val_data$price - full_fit) / val_data$price)*100
perf_mat[4,2] <- mean((val_data$price - full_fit_tfY1) / val_data$price)*100

# Result
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("Original", "Transformation_Y")
perf_mat