# Regression model predicting cold diagnostic count by weather and search volume in SNS

## Set the working directory
setwd("C:/Users/user/Desktop/drive/kuai/2_Analysis/5_KSB/workspace")

## Multivariate linear regression
cold <- read.csv("cold.csv")
str(cold)

# Check all NA
library(dplyr)
colSums(is.na(cold))


## Indices for the activated input variables
cold$요일 <- as.factor(cold$요일)
nCold <- dim(cold)[1]


## Split the data into the training/validation sets
trn_idx <- sample(1:nCold, round(0.7*nCold))
trn_data <- cold[trn_idx,]
val_data <- cold[-trn_idx,]


## Train the MLR
full_model <- lm(진료건수 ~., data = trn_data)
summary(full_model)


## Plot the result
# residual plot
plot(fitted(full_model), residuals(full_model), xlab = "Fitted values",
     ylab = "Residuals", main = "residual plot")
abline(0, 0, lty = 3)

# box plot
boxplot(residuals(full_model), ylab = "Residuals", main = "Box plot")

# normal probability plot
qqnorm(rstandard(full_model), ylab = "Standardized residuals", 
       xlab = "Normal scores", main = "Normal Q-Q")
abline(0, 1, col = "red")

# fitted value & real value
plot(trn_data$진료건수, fitted(full_model),
     xlab = "Real value", ylab = "Fitted value")
abline(0, 1, lty = 3)


## Prediction
options(warn = -1)
full_fit <- predict(full_model, newdata = val_data)

matplot(cbind(full_fit, val_data$진료건수), pch = 1, type = "b",
        col = c(1:2), ylab = "", xlab = "Observation")
legend("topright", legend = c("fitted value", "real value"), 
       pch = 1, col = c(1:2), bg = "white")


## Make upperbound formula
tmp_x <- paste(colnames(trn_data)[-1], collapse = "+")
tmp_xy <- paste("진료건수 ~", tmp_x, collapse = "")
as.formula(tmp_xy)


## Choosing variable 1 (Forward selection)
forward_model <- step(lm(진료건수 ~ 1, data = trn_data),
                      scope = list(upper = tmp_xy, lower = 진료건수 ~ 1),
                      direction = "forward", trace = 0)
summary(forward_model)


## Choosing variable 2  (Backward elimination)
backward_model <- step(full_model, 
                       scope = list(upper = tmp_xy, lower = 진료건수 ~ 1), 
                       direction = "backward", trace = 0)
summary(backward_model)


## Choosing variable 3 (Stepwise selection)
stepwise_model <- step(lm(진료건수 ~ 1, data = trn_data),
                       scope = list(upper = tmp_xy, lower = 진료건수 ~ 1),
                       direction = "both", trace = 0)
summary(stepwise_model)


## Compare all models by validation data
full_fit <- predict(full_model, newdata = val_data)
forward_fit <- predict(forward_model, newdata = val_data)
backward_fit <- predict(backward_model, newdata = val_data)
stepwise_fit <- predict(stepwise_model, newdata = val_data)


## Performance matrix
# 1. Mean squared error (MSE)
perf_mat <- matrix(0, 4, 4)
perf_mat[1,1] <- mean((val_data$진료건수 - full_fit)^2)
perf_mat[1,2] <- mean((val_data$진료건수 - forward_fit)^2)
perf_mat[1,3] <- mean((val_data$진료건수 - backward_fit)^2)
perf_mat[1,4] <- mean((val_data$진료건수 - stepwise_fit)^2)

# 2. Root mean squared error (RMSE)
perf_mat[2,1] <- sqrt(mean((val_data$진료건수 - full_fit)^2))
perf_mat[2,2] <- sqrt(mean((val_data$진료건수 - forward_fit)^2))
perf_mat[2,3] <- sqrt(mean((val_data$진료건수 - backward_fit)^2))
perf_mat[2,4] <- sqrt(mean((val_data$진료건수 - stepwise_fit)^2))

# 3. Mean absolute error (MAE)
perf_mat[3,1] <- mean(abs(val_data$진료건수 - full_fit))
perf_mat[3,2] <- mean(abs(val_data$진료건수 - forward_fit))
perf_mat[3,3] <- mean(abs(val_data$진료건수 - backward_fit))
perf_mat[3,4] <- mean(abs(val_data$진료건수 - stepwise_fit))

# 4. Mean absolute percentage error (MAPE)
perf_mat[4,1] <- mean((val_data$진료건수 - full_fit))*100
perf_mat[4,2] <- mean((val_data$진료건수 - forward_fit))*100
perf_mat[4,3] <- mean((val_data$진료건수 - backward_fit))*100
perf_mat[4,4] <- mean((val_data$진료건수 - stepwise_fit))*100

# Result
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("All", "Forward", "Backward", "Stepwise")
perf_mat