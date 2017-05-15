
######################################################################################
# supplementaryCode.R                                                                #
#    This code demonstrates how to use the following methods for the dataset "Meuse" #
#        the dataset "Meuse".                                                        #
#       1) Methods: random forests, boosted regression trees, weighted-KNN,          #
#                   cubist, generalised linear (GLM) and additive (GAM) models       #
#       2) Performing repeated k-fold cross-validation                               #
#       3) Performing sensitivity analyses on model parameters (if needed)           #
#       4) Mapping predicted values (example with random forests)                    #
#                                                                                    #
# Author: Julien Beguin <julien.beguin@canada.ca> (2016)                             #
######################################################################################


#######################
# Part 1 --           #
#     Load R packages #
#######################

# these packages must be installed in your R environment before running the code                                                           
require(sp)                                                           
require(caret) 
require(randomForest) 
require(gbm)
require(kknn)
require(Cubist) 

###############################
# Part 2 --                   #
#     Load the Meuse data set #
###############################

demo(meuse, echo=FALSE)
meuse.df <- meuse@data # data.frame with the sample data
meuse.df$SampleId <- rownames(meuse.df) # add unique sampleId to the 'meuse.df' data.frame
sampleCoord <- as.data.frame(meuse.grid@coords[as.numeric(rownames(meuse@data)),]) # select geographic coordinates associated with samples
sampleCoord$SampleId <- rownames(sampleCoord) # add sampleId to the 'sampleCoord' data.frame
meuse.df <- merge(meuse.df, sampleCoord, by = "SampleId")  # add geographic coordinates to the 'meuse.df' data.frame

#str(meuse.df) # see the structure of the data.frame
#str(meuse.grid) # see the structure of the SpatialPixelsDataFrame

## For validation: split the data into training vs validation datasets
## And replace 'data = meuse.df' by 'data = train_meuse.df' in the code below 
#smp_size <- floor(0.9 * nrow(meuse.df )) # set the percentage of the data to keep for training, here 90%
#train_ind <- sample(seq_len(nrow(meuse.df)), size = smp_size)
#train_meuse.df <- meuse.df[train_ind, ]
#test_meuse.df <- meuse.df[-train_ind, ]

###################################
# Part 3 --                       #
#     Set formula for each model  #
###################################
    
formula_CovariatesOnly <- as.formula(paste('log1p(om)', paste('~ dist + ffreq')))
formula_SpatialOnly <- as.formula(paste('log1p(om)', paste('~ x + y')))
formula_CovSpatial <- as.formula(paste('log1p(om)', paste('~ x + y + dist + ffreq')))

#############################################################################
# Part 4 --                                                                 #
#     Fit each model (covariates only, spatial only, covariates + spatial)  #
#     for every statistical method using 10-folds cross-validation repeated #
#     20 times                                                              #
#############################################################################

### set k-fold properties (number of folds & number of repetitions)
#set.seed(1) ## set seed if needed
nfolds = 10 ## define the number of folds 
repeats = 20 ## define the number of repetitions 
ctrlCV <- trainControl(method = "repeatedcv", number = nfolds, repeats = repeats, savePredictions = TRUE) # use repeated 10-folds crossvalidation and save predictions 

###########################
###- 1. Random Forests -###
###########################
CovariatesOnly_RF_10fCV <- train(formula_CovariatesOnly, data = meuse.df, na.action = na.omit, method = "rf", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(mtry = 2))
SpatialOnly_RF_10fCV <- train(formula_SpatialOnly, data = meuse.df, na.action = na.omit, method = "rf", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(mtry = 2))
CovSpatial_RF_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "rf", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(mtry = 2))

CovariatesOnly_RF_10fCV$results
SpatialOnly_RF_10fCV$results
CovSpatial_RF_10fCV$results

#- sensitivity analyses (be careful to select sensitive parameters; be patient, it takes more time to run)
RFgrid <- expand.grid(.mtry = c(2,3,4)) # mtry = the number of randomly selected predictors, k, to choose from at each split.
#CovSpatial_RF_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "rf", metric="RMSE", trControl = ctrlCV, tuneGrid = RFgrid)

#####################################
###- 2. Boosted Regression Trees -###
#####################################
CovariatesOnly_BRT_10fCV <- train(formula_CovariatesOnly, data = meuse.df, na.action = na.omit, method = "gbm", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(interaction.depth = 3, n.trees = 5000, shrinkage = 0.01, n.minobsinnode = 10), verbose = FALSE)
SpatialOnly_BRT_10fCV <- train(formula_SpatialOnly, data = meuse.df, na.action = na.omit, method = "gbm", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(interaction.depth = 3, n.trees = 5000, shrinkage = 0.01, n.minobsinnode = 10), verbose = FALSE)
CovSpatial_BRT_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "gbm", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(interaction.depth = 3, n.trees = 5000, shrinkage = 0.01, n.minobsinnode = 10), verbose = FALSE)

CovariatesOnly_BRT_10fCV$results
SpatialOnly_BRT_10fCV$results
CovSpatial_BRT_10fCV$results

#- sensitivity analyses (be careful to select sensitive parameters; be patient, it takes more time to run)
BRTgrid <- expand.grid(.interaction.depth = c(1,2,3,4), .n.trees = 5000, .shrinkage = c(0.001,0.01,0.05), .n.minobsinnode = c(5,10,20))
#CovSpatial_BRT_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "gbm", metric="RMSE", trControl = ctrlCV, tuneGrid = BRTgrid, verbose = FALSE)

#################
###- 3. KKNN -###
#################
CovariatesOnly_KKNN_10fCV <- train(formula_CovariatesOnly, data = meuse.df, na.action = na.omit, method = "kknn", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(kmax = 5, distance = 1, kernel = c("inv")))
SpatialOnly_KKNN_10fCV <- train(formula_SpatialOnly, data = meuse.df, na.action = na.omit, method = "kknn", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(kmax = 5, distance = 1, kernel = c("inv")))
CovSpatial_KKNN_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "kknn", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(kmax = 5, distance = 1, kernel = c("inv")))

CovariatesOnly_KKNN_10fCV$results
SpatialOnly_KKNN_10fCV$results
CovSpatial_KKNN_10fCV$results 

#- sensitivity analyses (be careful to select sensitive parameters; be patient, it takes more time to run)
KKNNgrid <- expand.grid(.kmax = c(5,10,20), .distance = c(1:2), .kernel = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"))
#CovSpatial_KKNN_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "kknn", metric="RMSE", trControl = ctrlCV, tuneGrid = KKNNgrid)

###################
###- 4. CUBIST -###
###################
CovariatesOnly_CUB_10fCV <- train(formula_CovariatesOnly, data = meuse.df, na.action = na.omit, method = "cubist", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(committees = c(2), neighbors = c(3)))
SpatialOnly_CUB_10fCV <- train(formula_SpatialOnly, data = meuse.df, na.action = na.omit, method = "cubist", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(committees = c(2), neighbors = c(3)))
CovSpatial_CUB_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "cubist", metric="RMSE", trControl = ctrlCV, tuneGrid = data.frame(committees = c(2), neighbors = c(3)))

CovariatesOnly_CUB_10fCV$results
SpatialOnly_CUB_10fCV$results
CovSpatial_CUB_10fCV$results

#- sensitivity analyses (be careful to select sensitive parameters; be patient, it takes more time to run)
Cubistgrid <- expand.grid(.committees = c(2,5,10,15,20,25,30), .neighbors = c(3,6,9))
#CovSpatial_CUB_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "cubist", metric="RMSE", trControl = ctrlCV, tuneGrid = Cubistgrid)

################
###- 5. GAM -###
################
CovariatesOnly_GAM_10fCV <- train(formula_CovariatesOnly, data = meuse.df, na.action = na.omit, method = "gam", metric="RMSE", trControl = ctrlCV)
SpatialOnly_GAM_10fCV <- train(formula_SpatialOnly, data = meuse.df, na.action = na.omit, method = "gam", metric="RMSE", trControl = ctrlCV)
CovSpatial_GAM_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "gam", metric="RMSE", trControl = ctrlCV)

CovariatesOnly_GAM_10fCV$results
SpatialOnly_GAM_10fCV$results
CovSpatial_GAM_10fCV$results

################
###- 6. GLM -###
################
CovariatesOnly_GLM_10fCV <- train(formula_CovariatesOnly, data = meuse.df, na.action = na.omit, method = "lm", metric="RMSE", trControl = ctrlCV)
SpatialOnly_GLM_10fCV <- train(formula_SpatialOnly, data = meuse.df, na.action = na.omit, method = "lm", metric="RMSE", trControl = ctrlCV)
CovSpatial_GLM_10fCV <- train(formula_CovSpatial, data = meuse.df, na.action = na.omit, method = "lm", metric="RMSE", trControl = ctrlCV)

CovariatesOnly_GLM_10fCV$results
SpatialOnly_GLM_10fCV$results
CovSpatial_GLM_10fCV$results

###########################
# Part 5 --               #
#     Extract the results #
###########################

#- Random Forests
Stat_CovOnly_RF <- CovariatesOnly_RF_10fCV$resample
Stat_SpatialOnly_RF <- SpatialOnly_RF_10fCV$resample
Stat_CovSpatial_RF <- CovSpatial_RF_10fCV$resample

#- Boosted Regression Trees
Stat_CovOnly_BRT <- CovariatesOnly_BRT_10fCV$resample
Stat_SpatialOnly_BRT <- SpatialOnly_BRT_10fCV$resample
Stat_CovSpatial_BRT <- CovSpatial_BRT_10fCV$resample

#- KKNN
Stat_CovOnly_KKNN <- CovariatesOnly_KKNN_10fCV$resample
Stat_SpatialOnly_KKNN <- SpatialOnly_KKNN_10fCV$resample
Stat_CovSpatial_KKNN <- CovSpatial_KKNN_10fCV$resample

#- CUBIST
Stat_CovOnly_CUB <- CovariatesOnly_CUB_10fCV$resample
Stat_SpatialOnly_CUB <- SpatialOnly_CUB_10fCV$resample
Stat_CovSpatial_CUB <- CovSpatial_CUB_10fCV$resample

#- GAM
Stat_CovOnly_GAM <- CovariatesOnly_GAM_10fCV$resample
Stat_SpatialOnly_GAM <- SpatialOnly_GAM_10fCV$resample
Stat_CovSpatial_GAM <- CovSpatial_GAM_10fCV$resample

#- GLM
Stat_CovOnly_GLM <- CovariatesOnly_GLM_10fCV$resample
Stat_SpatialOnly_GLM <- SpatialOnly_GLM_10fCV$resample
Stat_CovSpatial_GLM <- CovSpatial_GLM_10fCV$resample

rep <- as.factor(substr(Stat_CovOnly_RF$Resample, 8, 13))

#####################
# Part 6 --         #
#     Print results #
#####################

#-- Pseudo-RSquare

  q.ns.RF = quantile((aggregate(Stat_CovOnly_RF$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.RF = quantile((aggregate(Stat_SpatialOnly_RF$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.RF = quantile((aggregate(Stat_CovSpatial_RF$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.BRT = quantile((aggregate(Stat_CovOnly_BRT$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.BRT = quantile((aggregate(Stat_SpatialOnly_BRT$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.BRT = quantile((aggregate(Stat_CovSpatial_BRT$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.KKNN = quantile((aggregate(Stat_CovOnly_KKNN$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.KKNN = quantile((aggregate(Stat_SpatialOnly_KKNN$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.KKNN = quantile((aggregate(Stat_CovSpatial_KKNN$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.CUB = quantile((aggregate(Stat_CovOnly_CUB$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.CUB = quantile((aggregate(Stat_SpatialOnly_CUB$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.CUB = quantile((aggregate(Stat_CovSpatial_CUB$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.GAM = quantile((aggregate(Stat_CovOnly_GAM$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.GAM = quantile((aggregate(Stat_SpatialOnly_GAM$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.GAM = quantile((aggregate(Stat_CovSpatial_GAM$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975), na.rm=TRUE)
  q.ns.GLM = quantile((aggregate(Stat_CovOnly_GLM$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.GLM = quantile((aggregate(Stat_SpatialOnly_GLM$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.GLM = quantile((aggregate(Stat_CovSpatial_GLM$Rsquared, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))

pseudoR2 <- rbind(q.ns.RF,q.os.RF,q.s.RF,q.ns.BRT,q.os.BRT,q.s.BRT,q.ns.KKNN,q.os.KKNN,q.s.KKNN,q.ns.CUB,q.os.CUB,q.s.CUB,q.ns.GAM,q.os.GAM,q.s.GAM,q.ns.GLM,q.os.GLM,q.s.GLM)
method <- c(rep("RF", 3), rep("BRT", 3), rep("KKNN", 3), rep("CUB", 3), rep("GAM", 3), rep("GLM", 3))
variable <- rep("log1p(om)", 18)
model <- rep(c("non-spatial","spatial only","cov + spatial"), 6)
stat <- rep("Rsquared", 18)
pseudoR2table <- data.frame(cbind(variable, method, model, stat, pseudoR2))

#-- RSquare (example for ‘CovariatesOnly_RF_10fCV’ – just change the model name to get RSquare values for the other models/methods)
data_stat <- CovariatesOnly_RF_10fCV$pred
data_stat$fold <- substr(data_stat$Resample,1,6)
data_stat$rep <- substr(data_stat$Resample,8,12)
data_stat
data_stat$SqErr <- (CovariatesOnly_RF_10fCV$pred$obs - CovariatesOnly_RF_10fCV$pred$pred)^2
MSE <- aggregate(SqErr~fold+rep, data = data_stat, FUN = sum)
data_merge <- merge(data_stat,aggregate(obs ~fold+rep, data = data_stat, FUN = mean), by=c('fold','rep'))
data_merge$diffmean <- (data_merge$obs.x - data_merge$obs.y)^2
SST <- aggregate(diffmean~fold+rep, data = data_merge, FUN = sum)
R2 <- 1-MSE$SqErr/SST$diffmean
data.frame(cbind(MSE$fold, MSE$rep, R2))
quantile((aggregate(as.numeric(as.character(data.frame(cbind(MSE$fold, MSE$rep, R2))$R2)), list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))


#-- RMSE
  q.ns.RF = quantile((aggregate(Stat_CovOnly_RF$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.RF = quantile((aggregate(Stat_SpatialOnly_RF$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.RF = quantile((aggregate(Stat_CovSpatial_RF$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.BRT = quantile((aggregate(Stat_CovOnly_BRT$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.BRT = quantile((aggregate(Stat_SpatialOnly_BRT$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.BRT = quantile((aggregate(Stat_CovSpatial_BRT$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.KKNN = quantile((aggregate(Stat_CovOnly_KKNN$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.KKNN = quantile((aggregate(Stat_SpatialOnly_KKNN$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.KKNN = quantile((aggregate(Stat_CovSpatial_KKNN$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.CUB = quantile((aggregate(Stat_CovOnly_CUB$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.CUB = quantile((aggregate(Stat_SpatialOnly_CUB$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.CUB = quantile((aggregate(Stat_CovSpatial_CUB$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.GAM = quantile((aggregate(Stat_CovOnly_GAM$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.GAM = quantile((aggregate(Stat_SpatialOnly_GAM$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.GAM = quantile((aggregate(Stat_CovSpatial_GAM$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.ns.GLM = quantile((aggregate(Stat_CovOnly_GLM$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.os.GLM = quantile((aggregate(Stat_SpatialOnly_GLM$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))
  q.s.GLM = quantile((aggregate(Stat_CovSpatial_GLM$RMSE, list(rep), FUN=mean)$x), probs = c(0.025, 0.5, 0.975))

RMSE <- rbind(q.ns.RF,q.os.RF,q.s.RF,q.ns.BRT,q.os.BRT,q.s.BRT,q.ns.KKNN,q.os.KKNN,q.s.KKNN,q.ns.CUB,q.os.CUB,q.s.CUB,q.ns.GAM,q.os.GAM,q.s.GAM,q.ns.GLM,q.os.GLM,q.s.GLM)
method <- c(rep("RF", 3), rep("BRT", 3), rep("KKNN", 3), rep("CUB", 3), rep("GAM", 3), rep("GLM", 3))
variable <- rep("log1p(om)", 18)
model <- rep(c("non-spatial","spatial only","cov + spatial"), 6)
stat <- rep("RMSE", 18)
RMSEtable <- data.frame(cbind(variable, method, model, stat, RMSE))
#RMSEtable

data.frame(rbind(pseudoR2table, RMSEtable)) # print final results

#####################################
# Part 7 --                         #
#     Spatial predictions (mapping) #
#     ex: random forests            #
#####################################
formula_CovariatesOnly <- as.formula(paste('log1p(om)', paste('~ dist + ffreq')))
formula_SpatialOnly <- as.formula(paste('log1p(om)', paste('~ x + y')))
formula_CovSpatial <- as.formula(paste('log1p(om)', paste('~ x + y + dist + ffreq')))

modelFit <- randomForest(formula_CovSpatial, data = na.omit(meuse.df), mtry = 2, ntree = 5000, nodesize = 5)
prediction <- predict(modelFit, meuse.grid)
meuse.grid$mean_pred <- prediction
spplot(meuse.grid[c("mean_pred")], names.attr=c("randomForest"), sp.layout=list("sp.points", meuse, pch="+", cex=1.5, col="black")) 
