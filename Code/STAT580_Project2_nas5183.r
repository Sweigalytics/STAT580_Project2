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
exterior_cols = c("ExteriorMetalSd","ExteriorOtherSd","ExteriorVinylSd")
lot_cols = c("LotType","LotShape","LotFR2","LotFR3")

df_neighborhoods_separate <- df_neighborhoods %>% separate(Exterior, exterior_cols, sep=";") %>% separate(LotInfo, lot_cols, sep=";")
df_neighborhoods_separate_test <- df_neighborhoods_test %>% separate(Exterior, exterior_cols, sep=";") %>% separate(LotInfo, lot_cols, sep=";")

df_neighborhoods_separate[,c("LotFR2","LotFR3")] <- sapply(df_neighborhoods_separate[,c("LotFR2","LotFR3")], as.integer)
df_neighborhoods_separate_test[,c("LotFR2","LotFR3")] <- sapply(df_neighborhoods_separate_test[,c("LotFR2","LotFR3")], as.integer)


# Fill in "NA" for empty strings in `BsmtQual`, `BsmtFinType1`, and `GarageType`.
# Also replace the empty `LotFR3` columns with 0. We will assume they do not have frontage on 3 sides.
empty_string_cols <- c("BsmtQual","BsmtFinType1","GarageType")
df_neighborhoods_impute <- df_neighborhoods_separate %>% 
                            mutate_at(empty_string_cols, ~replace(., . == "", "NA")) %>%
                            mutate_at("LotFR3", ~replace(., is.na(.), 0))
df_neighborhoods_impute_test <- df_neighborhoods_separate_test %>% 
                                  mutate_at(empty_string_cols, ~replace(., . == "", "NA")) %>%
                                  mutate_at("LotFR3", ~replace(., is.na(.), 0))



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
  plot_LotFR2,
  plot_LotFR3,
  plot_OpenPorchSF,
  plot_OverallCond,
  plot_OverallQual,
  plot_SalePrice,
  plot_TotRmsAbvGrd,
  plot_WoodDeckSF,
  plot_YearBuilt,
  plot_YrSold
)
