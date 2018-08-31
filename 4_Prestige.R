# Regression model predicting job prestige

library(dummies) # dummy.data.frame
library(car) # scatterplotMatrix

## Set the working directory
setwd("C:/Users/user/Desktop/drive/kuai/2_Analysis/5_KSB/workspace")

## Multivariate linear regression
Prestige <- read.csv("Prestige.csv")
str(Prestige)

Prestige1 <- Prestige[,-1]
str(Prestige1)

# Check all NA
library(dplyr)
colSums(is.na(Prestige1))


## Indices for the activated input variables
nObs <- dim(Prestige1)[1]


## 1-c coding
dat <- dummy.data.frame(Prestige1, names = c("type"))
str(dat)


## Split the data into the training/validation sets
set.seed(2018040318)
trn_idx <- sample(1:nObs, round(0.7*nObs))
trn_data <- dat[trn_idx,]
val_data <- dat[-trn_idx,]
dim(trn_data)
dim(val_data)


## Train the MLR
full_model <- lm(income ~., data = trn_data)
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
scatterplotMatrix(~ income + prestige + education + women, data = dat, 
                  diagonal = "histogram", reg.line = FALSE, smoother = FALSE)
scatterplotMatrix(~ log(income) + prestige + education + women, data = dat, 
                  diagonal = "histogram", reg.line = FALSE, smoother = FALSE)

# Train the MLR
full_model_tfY <- lm(log(income) ~., data = trn_data)
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

matplot(cbind(val_data$income, full_fit, full_fit_tfY1), 
        pch = c(1:3), type = "b", col = c(1:3), ylab = "", xlab = "observation")
legend("topright", pch = c(1:3), col = c(1:3), bg = "white", 
       legend = c("real value", "predicted value", "predicted value (logY)"))


## Performance matrix
# 1. Mean squared error (MSE)
perf_mat <- matrix(0, 4, 2)
perf_mat[1,1] <- mean((val_data$income - full_fit)^2)
perf_mat[1,2] <- mean((val_data$income - full_fit_tfY1)^2)

# 2. Root mean squared error (RMSE)
perf_mat[2,1] <- sqrt(mean((val_data$income - full_fit)^2))
perf_mat[2,2] <- sqrt(mean((val_data$income - full_fit_tfY1)^2))

# 3. Mean absolute error (MAE)
perf_mat[3,1] <- mean(abs(val_data$income - full_fit))
perf_mat[3,2] <- mean(abs(val_data$income - full_fit_tfY1))

# 4. Mean absolute percentage error (MAPE)
perf_mat[4,1] <- mean((val_data$income - full_fit) / val_data$income)*100
perf_mat[4,2] <- mean((val_data$income - full_fit_tfY1) / val_data$income)*100

# Result
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("Original", "Transformation_Y")
perf_mat