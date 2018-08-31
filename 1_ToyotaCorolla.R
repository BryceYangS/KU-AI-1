# Regression model predicting Toyota used car price by age, distance driven, etc.

## Set the working directory
setwd("C:/Users/user/Desktop/drive/kuai/2_Analysis/5_KSB/workspace")

## Multivariate linear regression
corolla <- read.csv("ToyotaCorolla.csv")
corolla$Automatic <- as.factor(corolla$Automatic)

# Age from August 2004
# Distance driven by km
# Fuel type (gasoline, diesel, CNG)
# Horsepower
# Whether a car has Automatic transmission (yes is 1, no is 0)
# Volume of cylinder by cc
# Travel tax quarterly
# Weight by kg
# Price by EUR
str(corolla)

# Check all NA
library(dplyr)
colSums(is.na(corolla))


## Indices for the activated input variables
nCar <- dim(corolla)[1]
nVar <- dim(corolla)[2]

## Categorical variable -> 1-of-c coding
dummy_p <- rep(0, nCar)
dummy_d <- rep(0, nCar)
dummy_c <- rep(0, nCar)

p_idx <- which(corolla$Fuel_Type == "Petrol")
d_idx <- which(corolla$Fuel_Type == "Diesel")
c_idx <- which(corolla$Fuel_Type == "CNG")

dummy_p[p_idx] <- 1
dummy_d[d_idx] <- 1
dummy_c[c_idx] <- 1

Fuel <- data.frame(dummy_p, dummy_d, dummy_c)
names(Fuel) <- c("Petrol", "Diesel", "CNG")


## Prepare the data for MLR
mlr_data <- cbind(Fuel, corolla [,-3])
str(mlr_data)


## Split the data into the training/validation sets
trn_idx <- sample(1:nCar, round(0.7*nCar))
trn_data <- mlr_data[trn_idx,]
val_data <- mlr_data[-trn_idx,]

dim(trn_data)
dim(val_data)


## Train the MLR
full_model <- lm(Price ~., data = trn_data)
summary(full_model)


## Plot the result
# residual plot
plot(fitted(full_model), residuals(full_model), xlab = "Fitted values",
     ylab = "Residuals", main = "Residual plot")
abline(0, 0, lty = 3)

# box plot
boxplot(residuals(full_model), ylab = "Residuals", main = "Box plot")

# normal probability plot
qqnorm(rstandard(full_model), ylab = "Standardized residuals", 
       xlab = "Normal scores", main = "Normal Q-Q")
abline(0, 1, col = "red")

# fitted value & real value
plot(trn_data$Price, fitted(full_model), xlim = c(4000, 35000), ylim = c(4000, 30000), 
     xlab = "Real value (Price)", ylab = "Fitted value")
abline(0, 1, lty = 3)


## Prediction
options(warn = -1)
full_fit <- predict(full_model, newdata = val_data)

matplot(cbind(full_fit, val_data$Price), pch = 1, type = "b",
        col = c(1:2), ylab = "", xlab = "Observation")
legend("topright", legend = c("fitted value", "real value"), 
       pch = 1, col = c(1:2), bg = "white")


## Make upperbound formula
tmp_x <- paste(colnames(trn_data)[-11], collapse = "+")
tmp_xy <- paste("Price ~", tmp_x)
as.formula(tmp_xy)


## Choosing variable 1 (Forward selection)
forward_model <- step(lm(Price ~ 1, data = trn_data),
                      scope = list(upper = tmp_xy, lower = Price ~ 1),
                      direction = "forward", trace = 0)
summary(forward_model)


## Choosing variable 2 (Backward elimination)
backward_model <- step(full_model, 
                       scope = list(upper = tmp_xy, lower = Price ~ 1), 
                       direction = "backward", trace = 0)
summary(backward_model)


## Choosing variable 3 (Stepwise selection)
stepwise_model <- step(lm(Price ~ 1, data = trn_data),
                       scope = list(upper = tmp_xy, lower = Price ~ 1),
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
perf_mat[1,1] <- mean((val_data$Price - full_fit)^2)
perf_mat[1,2] <- mean((val_data$Price - forward_fit)^2)
perf_mat[1,3] <- mean((val_data$Price - backward_fit)^2)
perf_mat[1,4] <- mean((val_data$Price - stepwise_fit)^2)

# 2. Root mean squared error (RMSE)
perf_mat[2,1] <- sqrt(mean((val_data$Price - full_fit)^2))
perf_mat[2,2] <- sqrt(mean((val_data$Price - forward_fit)^2))
perf_mat[2,3] <- sqrt(mean((val_data$Price - backward_fit)^2))
perf_mat[2,4] <- sqrt(mean((val_data$Price - stepwise_fit)^2))

# 3. Mean absolute error (MAE)
perf_mat[3,1] <- mean(abs(val_data$Price - full_fit))
perf_mat[3,2] <- mean(abs(val_data$Price - forward_fit))
perf_mat[3,3] <- mean(abs(val_data$Price - backward_fit))
perf_mat[3,4] <- mean(abs(val_data$Price - stepwise_fit))

# 4. Mean absolute percentage error (MAPE)
perf_mat[4,1] <- mean((val_data$Price - full_fit))*100
perf_mat[4,2] <- mean((val_data$Price - forward_fit))*100
perf_mat[4,3] <- mean((val_data$Price - backward_fit))*100
perf_mat[4,4] <- mean((val_data$Price - stepwise_fit))*100

# Result
rownames(perf_mat) <- c("MSE", "RMSE", "MAE", "MAPE")
colnames(perf_mat) <- c("All", "Forward", "Backward", "Stepwise")
perf_mat