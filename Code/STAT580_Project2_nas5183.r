library(caret) # For dummyVars()
library(e1071) # For SVR
library(glmnet) # For Ridge Regression and Lasso
library(ggpubr)
library(leaps) # For regsubsets()
library(tidyverse)

###############################################
# 1. Data Ingestion, Cleaning, and Feature Engineering
###############################################

## Read in the source data files.
CollegeCr <- read.csv('../Data/CollegeCr.csv')
CollegeCr_test <- read.csv('../Data/CollegeCr.test.csv')
Edwards <- read.csv('../Data/Edwards.csv')
Edwards_test <- read.csv('../Data/Edwards.test.csv')
OldTown <- read.csv('../Data/OldTown.csv')
OldTown_test <- read.csv('../Data/OldTown.test.csv')

## Find the common column names between the files to combine. We will only include columns that are present in all three datasets.
intersect_colnames <- sort(Reduce(intersect,list(colnames(CollegeCr),colnames(Edwards),colnames(OldTown))))
intersect_colnames_test <- sort(Reduce(intersect,list(colnames(CollegeCr_test),colnames(Edwards_test),colnames(OldTown_test))))

## Combine the neighborhoods into a common dataframe.
## Retain their neighborhood name into a variable named `Neighborhood`
df_neighborhoods <- rbind(CollegeCr[,intersect_colnames] %>% mutate(Neighborhood = "CollegeCr"),
      Edwards[,intersect_colnames] %>% mutate(Neighborhood = "Edwards")
      ,OldTown[,intersect_colnames] %>% mutate(Neighborhood = "OldTown"))

df_neighborhoods_test <- rbind(CollegeCr_test[,intersect_colnames_test] %>% mutate(Neighborhood = "CollegeCr"),
                               Edwards_test[,intersect_colnames_test] %>% mutate(Neighborhood = "Edwards")
                               ,OldTown_test[,intersect_colnames_test] %>% mutate(Neighborhood = "OldTown"))

## The training/test split between neighborhoods is slightly imbalanced.
nrow(CollegeCr) / (nrow(CollegeCr) + nrow(CollegeCr_test)) # 0.7945205
nrow(Edwards) / (nrow(Edwards) + nrow(Edwards_test)) # 0.8314607
nrow(OldTown) / (nrow(OldTown) + nrow(OldTown_test)) # 0.8018018


## Separate the columns with multiple delimited values (`Exterior` and `LotInfo`) into separate columns.
exterior_cols = c("Exterior1st","ExteriorQual","ExteriorCond")
lot_cols = c("LotConfig","LotShape","LotArea","LotFrontage")

df_neighborhoods_separate <- df_neighborhoods %>% separate(Exterior, exterior_cols, sep=";") %>% separate(LotInfo, lot_cols, sep=";")
df_neighborhoods_separate_test <- df_neighborhoods_test %>% separate(Exterior, exterior_cols, sep=";") %>% separate(LotInfo, lot_cols, sep=";")

df_neighborhoods_separate[,c("LotArea","LotFrontage")] <- sapply(df_neighborhoods_separate[,c("LotArea","LotFrontage")], as.integer)
df_neighborhoods_separate_test[,c("LotArea","LotFrontage")] <- sapply(df_neighborhoods_separate_test[,c("LotArea","LotFrontage")], as.integer)

## De-duplicating records (rows 115, 116, and 279 are duplicates). We only perform this for training data because we do not want to remove records from test data.
df_neighborhoods_dedupe <- unique(df_neighborhoods_separate)

## Fill in "NA" for empty strings in `BsmtQual`, `BsmtFinType1`, and `GarageType`.
## Also replace the empty `LotFR3` columns with 0. We will assume they do not have frontage on 3 sides.
empty_string_cols <- c("BsmtQual","BsmtFinType1","GarageType")
df_neighborhoods_impute <- df_neighborhoods_dedupe %>% 
                            mutate_at(empty_string_cols, ~replace(., . == "", "NA")) %>%
                            mutate_at("BsmtCond", ~replace(., is.na(.), "NA")) %>%
                            mutate_at("LotFrontage", ~replace(., is.na(.), 0)
                            )
df_neighborhoods_impute_test <- df_neighborhoods_separate_test %>% 
                                  mutate_at(empty_string_cols, ~replace(., . == "", "NA")) %>%
                                  mutate_at("BsmtCond", ~replace(., is.na(.), "NA")) %>%
                                  mutate_at("LotFrontage", ~replace(., is.na(.), 0)
                                  )


## Plot histograms for numeric columns
num_cols <- colnames(select_if(df_neighborhoods_impute, is.numeric))

for(i in num_cols){
  
  assign(paste('plot_',i,sep=""), 
         ggplot(data=df_neighborhoods_impute, aes_string(x=i)) + geom_histogram() + theme(text = element_text(size = 8))
  )
}

ggarrange(
  plot_BedroomAbvGr,
  plot_BsmtFinSF1,
  plot_Fireplaces,
  plot_FullBath,
  plot_GrLivArea,
  plot_HalfBath,
  plot_LotArea,
  plot_LotFrontage,
  plot_OpenPorchSF,
  plot_OverallCond,
  plot_OverallQual,
  plot_SalePrice,
  plot_TotRmsAbvGrd,
  plot_WoodDeckSF,
  plot_YearBuilt,
  plot_YrSold
)

## Removing the record with `YrSold` of 2001 because it is an error. Its `YrSold` is too many years prior to `YearBuilt`.
## Also removing the `Utilities` column because it only has one value and provides no predictive value.
df_neighborhoods_cleaned <- df_neighborhoods_impute[-which(df_neighborhoods_impute$YrSold == 2001) , !names(df_neighborhoods_impute) == "Utilities"]
df_neighborhoods_cleaned_test <- df_neighborhoods_impute_test[ , !names(df_neighborhoods_impute) == "Utilities"] # There is no `YrSold` == 2001 record to remove from the test data.

# Encodes the categorical variables for Linear Regression methods.
dmy <- dummyVars(" ~ .", data = df_neighborhoods_cleaned[, !names(df_neighborhoods_cleaned) == "SalePrice"])
df_neighborhoods_final <- cbind(data.frame(predict(dmy, newdata = df_neighborhoods_cleaned)), df_neighborhoods_cleaned$SalePrice)
names(df_neighborhoods_final)[names(df_neighborhoods_final) == "df_neighborhoods_cleaned$SalePrice"] <- "SalePrice"

set.seed(580)
spec = c(train = .75, validate = .25)

g = sample(cut(
  seq(nrow(df_neighborhoods_final)), 
  nrow(df_neighborhoods_final)*cumsum(c(0,spec)),
  labels = names(spec)
))

res = split(df_neighborhoods_final, g)
# Setting the validation dataframe
df_neighborhoods_final_validation <- res$validate
# Setting the training dataframe
df_neighborhoods_final_train <- res$train


df_neighborhoods_final_test <- data.frame(predict(dmy, newdata = df_neighborhoods_cleaned_test))


# Pulling a list of dummy variable column names so we can exclude them from standardization.
dummyVarNames <- names(df_neighborhoods_final)[!(names(df_neighborhoods_final) %in% names(df_neighborhoods_cleaned))]

# Creating scaled versions for Ridge Regression and Lasso, excluding the dummy variable names
df_neighborhoods_scaled <- cbind(df_neighborhoods_cleaned[, -which(names(df_neighborhoods_cleaned) %in% c('SalePrice'))] %>% mutate_if(is.numeric, scale) %>% mutate_if(is.character, as.factor), df_neighborhoods_cleaned$SalePrice)
names(df_neighborhoods_scaled)[names(df_neighborhoods_scaled) == "df_neighborhoods_cleaned$SalePrice"] <- "SalePrice"

set.seed(580)

g = sample(cut(
  seq(nrow(df_neighborhoods_scaled)), 
  nrow(df_neighborhoods_scaled)*cumsum(c(0,spec)),
  labels = names(spec)
))

res = split(df_neighborhoods_scaled, g)
# Setting the validation dataframe
df_neighborhoods_scaled_validation <- res$validate
# Setting the training dataframe
df_neighborhoods_scaled_train <- res$train

df_neighborhoods_scaled_test <- df_neighborhoods_cleaned_test[, -which(names(df_neighborhoods_cleaned_test) %in% c('SalePrice'))] %>% mutate_if(is.numeric, scale) %>% mutate_if(is.character, as.factor)

num_cols <- colnames(select_if(df_neighborhoods_scaled, is.numeric))

for(i in num_cols){
  
  assign(paste('plot_scaled_',i,sep=""), 
         ggplot(data=df_neighborhoods_scaled, aes_string(x=i)) + geom_histogram() + theme(text = element_text(size = 8))
  )
}

ggarrange(
  plot_scaled_BedroomAbvGr,
  plot_scaled_BsmtFinSF1,
  plot_scaled_Fireplaces,
  plot_scaled_FullBath,
  plot_scaled_GrLivArea,
  plot_scaled_HalfBath,
  plot_scaled_LotArea,
  plot_scaled_LotFrontage,
  plot_scaled_OpenPorchSF,
  plot_scaled_OverallCond,
  plot_scaled_OverallQual,
  plot_scaled_SalePrice,
  plot_scaled_TotRmsAbvGrd,
  plot_scaled_WoodDeckSF,
  plot_scaled_YearBuilt,
  plot_scaled_YrSold
)

# Not needed for now, but will find the names of non-numeric columns.
# non_num_cols <- colnames(select_if(df_neighborhoods_scaled, negate(is.numeric)))



#write.csv(df_neighborhoods_final, '../df_neighborhoods_final.csv')



# Saving this code in case we need to just change variables to factors instead of encoding.
# str(df_neighborhoods_scaled %>% mutate_if(negate(is.numeric), as.factor))


str(df_neighborhoods_scaled_test %>% mutate_if(negate(is.numeric), as.factor))


###############################################
# 2. Variable Selection and fitting
###############################################

## Forward Stepwise Selection
regit.fwd <- regsubsets(SalePrice~.,data=df_neighborhoods_final_train, nvmax=86, method="forward")
regit.fwd.summary <- summary(regit.fwd)

min_cp.fwd <- which.min(regit.fwd.summary$cp) # 22 Variables
min_bic.fwd <- which.min(regit.fwd.summary$bic) # 18 Variables
max_adjr2.fwd <- which.max(regit.fwd.summary$adjr2) # 34 Variables

plot(regit.fwd.summary$rss, xlab="Number of Variables", ylab="RSS", type="l")

plot(regit.fwd.summary$adjr2, xlab="Number of Variables", ylab="Adjusted RSq", type="l")
points(max_adjr2.fwd, regit.fwd.summary$adjr2[max_adjr2.fwd], col="red", cex=2, pch=20)

plot(regit.fwd.summary$cp, xlab="Number of Variables", ylab="Cp", type="l")
points(min_cp.fwd, regit.fwd.summary$cp[min_cp.fwd], col="red", cex=2, pch=20)

plot(regit.fwd.summary$bic, xlab="Number of Variables", ylab="BIC", type="l")
points(min_bic.fwd, regit.fwd.summary$bic[min_bic.fwd], col="red", cex=2, pch=20)

### Parsing the selected variables for each Cp, BIC, and Adj R^2 methods.
linear_forward_cp_vars <- paste(names(which(regit.fwd.summary$outmat[min_cp.fwd, ] == "*")), collapse="+")
linear_forward_bic_vars <- paste(names(which(regit.fwd.summary$outmat[min_bic.fwd, ] == "*")), collapse="+")
linear_forward_adjr2_vars <- paste(names(which(regit.fwd.summary$outmat[max_adjr2.fwd, ] == "*")), collapse="+")

### Fitting the linear models for the three selection criteria.
linear_forward_cp_model <- lm(paste("SalePrice ~ ",linear_forward_cp_vars,sep=""), data = df_neighborhoods_final_train)
linear_forward_bic_model <- lm(paste("SalePrice ~ ",linear_forward_bic_vars,sep=""), data = df_neighborhoods_final_train)
linear_forward_adjr2_model <- lm(paste("SalePrice ~ ",linear_forward_adjr2_vars,sep=""), data = df_neighborhoods_final_train)

### Storing the predictions for each model against the validation data.
linear_forward_cp_model.pred <- predict(linear_forward_cp_model, df_neighborhoods_final_validation)
linear_forward_bic_model.pred <- predict(linear_forward_bic_model, df_neighborhoods_final_validation)
linear_forward_adjr2_model.pred <- predict(linear_forward_adjr2_model, df_neighborhoods_final_validation)

#### Calculating MSE for each model
mse_linear_forward_cp_model <- c("Linear Forward Selected Cp",mean((linear_forward_cp_model.pred - df_neighborhoods_final_validation$SalePrice)^2)) # 1006790081
mse_linear_forward_bic_model <- c("Linear Forward Selected BIC",mean((linear_forward_bic_model.pred - df_neighborhoods_final_validation$SalePrice)^2)) # 956447763
mse_linear_forward_adjr2_model <- c("Linear Forward Selected Adj R^2",mean((linear_forward_adjr2_model.pred - df_neighborhoods_final_validation$SalePrice)^2)) # 1068484673


## Backward stepwise selection.
regit.bwd <- regsubsets(SalePrice~.,data=df_neighborhoods_final, nvmax=86, method="backward")
regit.bwd.summary <- summary(regit.bwd)

min_cp.bwd <- which.min(regit.bwd.summary$cp) # 20 Variables
min_bic.bwd <- which.min(regit.bwd.summary$bic) # 16 Variables
max_adjr2.bwd <- which.max(regit.bwd.summary$adjr2) # 37 Variables

plot(regit.bwd.summary$rss, xlab="Number of Variables", ylab="RSS", type="l")

plot(regit.bwd.summary$adjr2, xlab="Number of Variables", ylab="Adjusted RSq", type="l")
points(max_adjr2.bwd, regit.bwd.summary$adjr2[max_adjr2.bwd], col="red", cex=2, pch=20)

plot(regit.bwd.summary$cp, xlab="Number of Variables", ylab="Cp", type="l")
points(min_cp.bwd, regit.bwd.summary$cp[min_cp.bwd], col="red", cex=2, pch=20)

plot(regit.bwd.summary$bic, xlab="Number of Variables", ylab="BIC", type="l")
points(min_bic.bwd, regit.bwd.summary$bic[min_bic.bwd], col="red", cex=2, pch=20)

### Parsing the selected variables for each Cp, BIC, and Adj R^2 methods.
linear_backward_cp_vars <- paste(names(which(regit.bwd.summary$outmat[min_cp.bwd, ] == "*")), collapse="+")
linear_backward_bic_vars <- paste(names(which(regit.bwd.summary$outmat[min_bic.bwd, ] == "*")), collapse="+")
linear_backward_adjr2_vars <- paste(names(which(regit.bwd.summary$outmat[max_adjr2.bwd, ] == "*")), collapse="+")

### Fitting the linear models for the three selection criteria.
linear_backward_cp_model <- lm(paste("SalePrice ~ ",linear_backward_cp_vars,sep=""), data = df_neighborhoods_final_train)
linear_backward_bic_model <- lm(paste("SalePrice ~ ",linear_backward_bic_vars,sep=""), data = df_neighborhoods_final_train)
linear_backward_adjr2_model <- lm(paste("SalePrice ~ ",linear_backward_adjr2_vars,sep=""), data = df_neighborhoods_final_train)

### Storing the predictions for each model against the validation data.
linear_backward_cp_model.pred <- predict(linear_backward_cp_model, df_neighborhoods_final_validation)
linear_backward_bic_model.pred <- predict(linear_backward_bic_model, df_neighborhoods_final_validation)
linear_backward_adjr2_model.pred <- predict(linear_backward_adjr2_model, df_neighborhoods_final_validation)

### Calculating MSE for each model
mse_linear_backward_cp_model <- c("Linear Backward Selected Cp",mean((linear_backward_cp_model.pred - df_neighborhoods_final_validation$SalePrice)^2)) # 902472200
mse_linear_backward_bic_model <- c("Linear Backward Selected BIC",mean((linear_backward_bic_model.pred - df_neighborhoods_final_validation$SalePrice)^2)) # 867542533
mse_linear_backward_adjr2_model <- c("Linear Backward Selected Adj R^2",mean((linear_backward_adjr2_model.pred - df_neighborhoods_final_validation$SalePrice)^2)) # 915621549


## Ridge Regression
x.train <- model.matrix(SalePrice~.,df_neighborhoods_scaled_train)[,-1]
y.train <- df_neighborhoods_scaled_train$SalePrice

grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(x.train,y.train,alpha=0,lambda=grid)

set.seed(1)
cv.out=cv.glmnet(x.train,y.train,alpha=0)
plot(cv.out)

bestlam=cv.out$lambda.min
bestlam # 215004.4

out=glmnet(x.train,y.train,alpha=0)
predict(out,type="coefficients",s=bestlam)

x.validation <- model.matrix(SalePrice~.,df_neighborhoods_scaled_validation)[,-1]
y.validation <- df_neighborhoods_scaled_validation$SalePrice

ridge.pred <- predict(ridge.mod, s = bestlam, newx = x.validation)
mse_ridge_regression <- c("Ridge Regression",mean((ridge.pred - y.validation)^2)) # 1,548,617,820


# Lasso
x.train <- model.matrix(SalePrice~.,df_neighborhoods_scaled_train)[,-1]
y.train <- df_neighborhoods_scaled_train$SalePrice

lasso.mod=glmnet(x.train,y.train,alpha=1,lambda=grid)
plot(lasso.mod)

set.seed(1)
cv.out=cv.glmnet(x.train,y.train,alpha=1)
plot(cv.out)

bestlam=cv.out$lambda.min
bestlam # 15525.06

out=glmnet(x.train,y.train,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)
lasso.coef

x.validation <- model.matrix(SalePrice~.,df_neighborhoods_scaled_validation)[,-1]
y.validation <- df_neighborhoods_scaled_validation$SalePrice

lasso.pred <- predict(lasso.mod, s = bestlam, newx = x.validation)
mse_lasso <- c("Lasso",mean((lasso.pred - y.validation)^2)) # 1,800,027,642


# Elastic Net

## Found a working example here: http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/153-penalized-regression-essentials-ridge-lasso-elastic-net/#elastic-net

## Set Train Control
set.seed(580)
train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 5,
                              search = "random",
                              verboseIter = TRUE)

elastic_train_data <- cbind(x.train, y.train)
colnames(elastic_train_data)[colnames(elastic_train_data) == 'y.train'] <- 'SalePrice'

elastic_net_model <- train(SalePrice ~ .,
                           data = elastic_train_data,
                           method = "glmnet",
                           tuneLength = 25,
                           trControl = train_control)

x.validation <- model.matrix(SalePrice~.,df_neighborhoods_scaled_validation)[,-1]
y.validation <- df_neighborhoods_scaled_validation$SalePrice

elastic_net.pred <- elastic_net_model %>% predict(x.validation)
mse_elastic_net <- c("Elastic Net",mean((elastic_net.pred - y.validation)^2)) #983,506,727.699282

# Random Forest Regression
library(randomForest)
set.seed(580)

rf.neighborhoods <- randomForest(SalePrice~., data = df_neighborhoods_final_train, mtry=85, importance=TRUE, ntree=100)
rf.pred <- predict(rf.neighborhoods, newdata = df_neighborhoods_final_validation)
mse_random_forest <- mean((rf.pred - y.validation)^2) #703846773

importance(rf.neighborhoods)


###############################################
# Model Selection
###############################################

df_mse <- data.frame(rbind(mse_linear_forward_cp_model,
                mse_linear_forward_bic_model,
                mse_linear_forward_adjr2_model,
                mse_linear_backward_cp_model,
                mse_linear_backward_bic_model,
                mse_linear_backward_adjr2_model,
                mse_ridge_regression,
                mse_lasso,
                mse_elastic_net))

colnames(df_mse) <- c("Model", "Mean Squared Error")
df_mse$`Mean Squared Error` <- as.numeric(df_mse$`Mean Squared Error`)
df_mse %>% arrange(`Mean Squared Error`)
