library(caret) # For dummyVars()
library(ggpubr)
library(tidyverse)

# Read in the source data files.
CollegeCr <- read.csv('../Data/CollegeCr.csv')
CollegeCr_test <- read.csv('../Data/CollegeCr.test.csv')
Edwards <- read.csv('../Data/Edwards.csv')
Edwards_test <- read.csv('../Data/Edwards.test.csv')
OldTown <- read.csv('../Data/OldTown.csv')
OldTown_test <- read.csv('../Data/OldTown.test.csv')

# Find the common column names between the files to combine.
intersect_colnames <- sort(Reduce(intersect,list(colnames(CollegeCr),colnames(Edwards),colnames(OldTown))))
intersect_colnames_test <- sort(Reduce(intersect,list(colnames(CollegeCr_test),colnames(Edwards_test),colnames(OldTown_test))))

# Combine the neighborhoods into a common dataframe.
# Retain their neighborhood name into a variable named `Neighborhood`
df_neighborhoods <- rbind(CollegeCr[,intersect_colnames] %>% mutate(Neighborhood = "CollegeCr"),
      Edwards[,intersect_colnames] %>% mutate(Neighborhood = "Edwards")
      ,OldTown[,intersect_colnames] %>% mutate(Neighborhood = "OldTown"))

df_neighborhoods_test <- rbind(CollegeCr_test[,intersect_colnames_test] %>% mutate(Neighborhood = "CollegeCr"),
                               Edwards_test[,intersect_colnames_test] %>% mutate(Neighborhood = "Edwards")
                               ,OldTown_test[,intersect_colnames_test] %>% mutate(Neighborhood = "OldTown"))

# The training/test split between neighborhoods is slightly imbalanced.
nrow(CollegeCr) / (nrow(CollegeCr) + nrow(CollegeCr_test)) # 0.7945205
nrow(Edwards) / (nrow(Edwards) + nrow(Edwards_test)) # 0.8314607
nrow(OldTown) / (nrow(OldTown) + nrow(OldTown_test)) # 0.8018018


# Separate the columns with multiple delimited values (`Exterior` and `LotInfo`) into separate columns.
exterior_cols = c("Exterior1st","ExteriorQual","ExteriorCond")
lot_cols = c("LotConfig","LotShape","LotArea","LotFrontage")

df_neighborhoods_separate <- df_neighborhoods %>% separate(Exterior, exterior_cols, sep=";") %>% separate(LotInfo, lot_cols, sep=";")
df_neighborhoods_separate_test <- df_neighborhoods_test %>% separate(Exterior, exterior_cols, sep=";") %>% separate(LotInfo, lot_cols, sep=";")

df_neighborhoods_separate[,c("LotArea","LotFrontage")] <- sapply(df_neighborhoods_separate[,c("LotArea","LotFrontage")], as.integer)
df_neighborhoods_separate_test[,c("LotArea","LotFrontage")] <- sapply(df_neighborhoods_separate_test[,c("LotArea","LotFrontage")], as.integer)


# Fill in "NA" for empty strings in `BsmtQual`, `BsmtFinType1`, and `GarageType`.
# Also replace the empty `LotFR3` columns with 0. We will assume they do not have frontage on 3 sides.
empty_string_cols <- c("BsmtQual","BsmtFinType1","GarageType")
df_neighborhoods_impute <- df_neighborhoods_separate %>% 
                            mutate_at(empty_string_cols, ~replace(., . == "", "NA")) %>%
                            mutate_at("BsmtCond", ~replace(., is.na(.), "NA")) %>%
                            mutate_at("LotFrontage", ~replace(., is.na(.), 0)
                            )
df_neighborhoods_impute_test <- df_neighborhoods_separate_test %>% 
                                  mutate_at(empty_string_cols, ~replace(., . == "", "NA")) %>%
                                  mutate_at("BsmtCond", ~replace(., is.na(.), "NA")) %>%
                                  mutate_at("LotFrontage", ~replace(., is.na(.), 0)
                                  )


# Plot histograms for numeric columns
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

# Removing the record with `YrSold` of 2001 because it is an error.
# Also removing the `Utilities` column because it only has one value.
df_neighborhoods_cleaned <- df_neighborhoods_impute[-which(df_neighborhoods_impute$YrSold == 2001) , !names(df_neighborhoods_impute) == "Utilities"]
df_neighborhoods_cleaned_test <- df_neighborhoods_impute_test[ , !names(df_neighborhoods_impute) == "Utilities"] # There is no `YrSold` == 2001 record to remove from the test data.


df_neighborhoods_scaled <- df_neighborhoods_cleaned %>% mutate_if(is.numeric, scale)
df_neighborhoods_scaled_test <- df_neighborhoods_cleaned_test %>% mutate_if(is.numeric, scale)

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

# Encodes the categorical 
dmy <- dummyVars(" ~ .", data = df_neighborhoods_scaled[, !names(df_neighborhoods_scaled) == "SalePrice"])
df_neighborhoods_final <- cbind(data.frame(predict(dmy, newdata = df_neighborhoods_scaled)), df_neighborhoods_scaled$SalePrice)
names(df_neighborhoods_final)[names(df_neighborhoods_final) == "df_neighborhoods_scaled$SalePrice"] <- "SalePrice"
write.csv(df_neighborhoods_final, '../df_neighborhoods_final.csv')

df_neighborhoods_final_test <- data.frame(predict(dmy, newdata = df_neighborhoods_scaled_test))

# Saving this code in case we need to just change variables to factors instead of encoding.
# str(df_neighborhoods_scaled %>% mutate_if(negate(is.numeric), as.factor))


str(df_neighborhoods_scaled_test %>% mutate_if(negate(is.numeric), as.factor))
