## Define evaluation function
perf_eval <- function(cm){
  
  # Accuracy
  ACC = (cm[2, 2] + cm[1, 1]/sum(cm))
  
  # TPR, Sensitivity, Recall
  sensitivity = cm[2, 2] / sum(cm[2, 1] + cm[2, 2])
  
  # TNR, Specificity
  specificity = cm[1, 1] / sum(cm[1,])
  
  # Precision, Positive Predictive
  precision = cm[2, 2] / sum(cm[,2])
  
  # F1 score
  F1 = 2 * sensitivity * specificity / (sensitivity + precision)
  
  # BCR
  BCR = sqrt(sensitivity)
  
  # Save as dataframe
  re <- data.frame(sensitivity = sensitivity,
                   specificity = specificity,
                   precision = precision,
                   ACC = ACC,
                   BCR = BCR,
                   F1 = F1)
  
  return(re)
  
}


## Set the working directory
setwd("C:/Users/user/Desktop/drive/kuai/2_Analysis/5_KSB/workspace")


## Load and check data
dat <- read.csv("Wine.csv")
str(dat)

# Check all NA
library(dplyr)
colSums(is.na(dat))


## Categorical variable: factor
dat$quality <- as.factor(dat$quality)
table(dat$quality)


## Split train & test set
set.seed(2018040318)
train_index <- sample(1:nrow(dat), round(nrow(dat) * 0.7), replace = F)
dat_train <- dat[train_index,]
dat_test <- dat[-train_index,]


## Draw some plots
# boxplot: volatile.acidity ~ quality
boxplot(volatile.acidity ~ quality, data = dat, main = 'boxplot volatile.acidity',
        xlab = 'volatile.acidity', ylab = 'quality')

# boxplot: density ~ quality
boxplot(density ~ quality, data = dat, main = 'boxplot density',
        xlab = 'density', ylab = 'quality')


## Bar plot: class distribution
counts <- table(dat$quality)
barplot(counts, main = 'bar plot: class counts',
        xlab = 'quality', col = c('darkblue', 'red'))


## Build logistic regression model
# Fit model
model_full <- glm(quality~., dat_train, family = binomial())
summary(model_full)

# Get coefficients and p-value
summary_full <- summary(model_full)
coeffs_full <- summary_full$coefficients
coeffs_full

nvar_full <- nrow(coeffs_full) - 1 # -1 : store except intercept 
nvar_full


## Test (cutoff = 0.5)
pred_prob <- predict(model_full, dat_test, type = "response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.5] <- 1
cm <- table(actual = dat_test$quality, pred = pred_class)
perf_eval(cm)


## Forward selection 
model_fwd <- step(glm(quality ~ 1, dat_train, family = binomial()),
                  direction = 'forward', trace = 0,
                  scope = formula(model_full))
summary(model_fwd)

# Get coefficients and p-value
summary_fwd <- summary(model_fwd)
coeffs_fwd <- summary_fwd$coefficients
coeffs_fwd

nvar_fwd <- nrow(coeffs_fwd) - 1
nvar_fwd

setdiff(rownames(coeffs_full), rownames(coeffs_fwd))

# Test (cutoff = 0.5)
pred_prob <- predict(model_fwd, dat_test, type = "response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.5] <- 1
cm <- table(actual = dat_test$quality, pred = pred_class)
perf_eval(cm)


## Backward selection
model_bwd <- step(glm(quality ~., dat_train, family = binomial()),
                  direction = "backward",
                  scope = list(lower = quality~1,
                               upper = formula(model_full)))
summary(model_bwd)

# Get coefficients and p-value
summary_bwd <- summary(model_bwd)
coeffs_bwd <- summary_bwd$coefficients
coeffs_bwd

nvar_bwd <- nrow(coeffs_bwd) - 1
nvar_bwd

# Test (cutoff = 0.5)
pred_prob <- predict(model_bwd, dat_test, type = "response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.5] <- 1
cm <- table(actual = dat_test$quality, pred = pred_class)
perf_eval(cm)


## Stepwise selection
model_step <- step(glm(quality ~ 1, dat_train, family = binomial()),
                   direction = "both", trace = 0,
                   scope = list(lower = quality ~ 1,
                                upper = formula(model_full)))
summary(model_step)

# Get coefficients and p-value
summary_step <- summary(model_step)
coeffs_step <- summary_step$coefficients
coeffs_step

nvar_step <- nrow(coeffs_step) - 1
nvar_step

# Test (cutoff = 0.5)
pred_prob <- predict(model_step, dat_test, type = "response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.5] <- 1
cm <- table(actual = dat_test$quality, pred = pred_class)
perf_eval(cm)