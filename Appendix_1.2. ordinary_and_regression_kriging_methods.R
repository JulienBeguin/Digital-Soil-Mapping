
#########################################################################################
# supplementaryCode.R                                                                   #
#    This code demonstrates how to use the following methods for the dataset "Meuse"    #  
#                                                                                       #
#       1) Methods: local ordinary kriging, regression-kriging & random forests-kriging #                 
#       2) Fitting the model                                                            #
#       3) Performing repeated k-fold cross-validation                                  #
#       4) Mapping predicted values                                                     #
#                                                                                       #
# Author: Julien Beguin <julien.beguin@canada.ca> (2016)                                #
#         adapted from the code of Geir-Arne Fuglstad (see appendix 1.3)                #
#########################################################################################

#######################
# Part 1 --           #
#     Load R packages #
#######################

# these packages must be installed in your R environment before running the code                                                           
require(sp)
require(gstat)
require(GSIF)
 
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

#str(meuse.df) # see de structure of the data.frame
#str(meuse.grid) # see de structure of the data.frame

coordinates(meuse.df) =~ x + y
str(meuse.df)
meuse.df@proj4string <- meuse.grid@proj4string
meuse.df@bbox <- meuse.grid@bbox
str(meuse.df)

## If needed, split the data into 90%(testing)-10%(validation)
set.seed(1)
smp_size <- floor(0.9 * nrow(meuse.df))
train_ind <- sample(seq_len(nrow(meuse.df)), size = smp_size)
train_meuse.df <- meuse.df[train_ind, ]
test_meuse.df <- meuse.df[-train_ind, ]
# in this case, replace below "meuse.df" by "train_meuse.df"

############################################################################
# Part 3 --                                                                #
#     K-Fold Cross-Validation for ordinary kriging, regression-kriging and #
#     random forests-kriging                                               #
############################################################################

# Iterate through each repetition
 set.seed(1)
 numRep = 20
 numFold = 10      
 #response = train_meuse.df[which(is.na(train_meuse.df@data$om)== FALSE),] # for training-validation
 response = meuse.df[which(is.na(meuse.df@data$om)== FALSE),]
 
 formula_mod <- as.formula(paste('log1p(om)', paste('~ dist + ffreq')))
 mtry_value=2

 # Set up arrays to store MSE, SST, and PseudoR2
 MSE.RF  = array(0, dim = c(numRep, numFold))
 MSE.RK <- MSE.RF 
 MSE.OK <- MSE.RF
 SST = MSE.RF
 PseudoR2.RF <- MSE.RF
 PseudoR2.RK <- MSE.RF
 PseudoR2.OK <- MSE.RF
 # For loop to record results for each fold and repetition
 for(cRep in 1:numRep){

  # Randomly divide data into folds
  bPoints = floor(seq(1, 1+length(response), length.out = numFold+1))
  idxPerm = sample.int(length(response))
  
  # Iterate through each fold
  for(cFold in 1:numFold){
        cat("Current rep =", cRep, "with fold =", cFold, "\n")

      # Remove data in fold cFold
      idx = sort(idxPerm[bPoints[cFold]:(bPoints[cFold+1]-1)])
      fResponse = response

      # Calculate predictive square error of model with only intercept
      SST[cRep, cFold] = sum((log1p(fResponse@data[idx,]$om) - mean(log1p(fResponse@data[-idx,]$om),na.rm = TRUE))^2, na.rm = TRUE)

      # Fit models
                
        #- Local ordinary kriging         
        lzn.vgm = variogram(log1p(om) ~ 1, fResponse[-idx,])		  
        lzn.fit = fit.variogram(lzn.vgm, model = vgm("Exp"))
        lzn.kriged = krige(log1p(om) ~ 1, fResponse[-idx,], fResponse[idx,], model = lzn.fit, nmax = 10)

        #- Regression-kriging
        mod.fit.RK = fit.gstatModel(fResponse[-idx,], formula_mod, vgmFun = "Exp",  method = "GLM", meuse.grid)                                                
        pred.RK <- predict(mod.fit.RK, fResponse[idx,], nfold=5)
                  
        #- Random forests-kriging
         mod.fit.RF = fit.gstatModel(fResponse[-idx,], formula_mod, vgmFun = "Exp",  method = "randomForest", mtry=mtry_value, meuse.grid)                                                
         pred.RF <- predict(mod.fit.RF, fResponse[idx,], nfold=5)

         # Pseudo-Rsquared
         PseudoR2.OK[cRep, cFold] <- cor(log1p(fResponse@data[idx,]$om), lzn.kriged@data$var1.pred, use = "pairwise.complete.obs")^2                
         PseudoR2.RF[cRep, cFold] <- cor(log1p(fResponse@data[idx,]$om), log1p(pred.RF$predicted@data$var1.pred), use = "pairwise.complete.obs")^2
         PseudoR2.RK[cRep, cFold] <- cor(log1p(fResponse@data[idx,]$om), pred.RK$predicted@data$var1.pred, use = "pairwise.complete.obs")^2
  
         # Mean square prediction error
         MSE.OK[cRep, cFold] <- sum((log1p(fResponse@data[idx,]$om) - lzn.kriged@data$var1.pred)^2, na.rm = TRUE)              
         MSE.RF[cRep, cFold] <- sum((log1p(fResponse@data[idx,]$om) - log1p(pred.RF$predicted@data$var1.pred))^2, na.rm = TRUE)
         MSE.RK[cRep, cFold] <- sum((log1p(fResponse@data[idx,]$om) - pred.RK$predicted@data$var1.pred)^2, na.rm = TRUE)

       }
  }

  # Calculate median of R2 values for each repetition
  R2values.OK = 1-MSE.OK/SST
  R2median.OK = apply(R2values.OK, 1, median)  
  R2values.RF = 1-MSE.RF/SST
  R2median.RF = apply(R2values.RF, 1, median)
  R2values.RK = 1-MSE.RK/SST
  R2median.RK = apply(R2values.RK, 1, median)
  
  # Calculate median pseudoR2 values
  PseudoR2median.OK <- apply(PseudoR2.OK, 1, median)  
  PseudoR2median.RF <- apply(PseudoR2.RF, 1, median)
  PseudoR2median.RK <- apply(PseudoR2.RK, 1, median)

  # Calculate root mean predictive square error (%)
  RMSE.OK = (sqrt(apply(MSE.OK, 1, sum)/length(fResponse@data$om)))/mean(log1p(response@data$om))  
  RMSE.RF = (sqrt(apply(MSE.RF, 1, sum)/length(fResponse@data$om)))/ mean(log1p(response@data$om))
  RMSE.RK = (sqrt(apply(MSE.RK, 1, sum)/length(fResponse@data$om)))/ mean(log1p(response@data$om))

  ########################  
  ##- 5. Print results -##
  ########################
  
  #- quantiles
  q.R2.OK = quantile(R2median.OK, c(0.025,0.5,0.975))  
  q.R2.RK = quantile(R2median.RK, c(0.025,0.5,0.975))
  q.R2.RF = quantile(R2median.RF, c(0.025,0.5,0.975))
  q.PseudoR2.OK = quantile(PseudoR2median.OK, c(0.025,0.5,0.975))  
  q.PseudoR2.RK = quantile(PseudoR2median.RK, c(0.025,0.5,0.975))
  q.PseudoR2.RF = quantile(PseudoR2median.RF, c(0.025,0.5,0.975))
  q.RMSE.OK = quantile(RMSE.OK, c(0.025,0.5,0.975))  
  q.RMSE.RK = quantile(RMSE.RK, c(0.025,0.5,0.975))
  q.RMSE.RF = quantile(RMSE.RF, c(0.025,0.5,0.975))
 
  #- print values
  q.R2.OK   
  q.R2.RK 
  q.R2.RF 
  q.PseudoR2.OK 
  q.PseudoR2.RK 
  q.PseudoR2.RF 
  q.RMSE.OK  
  q.RMSE.RK 
  q.RMSE.RF 

  values <- rbind(q.R2.OK, q.R2.RK, q.R2.RF, q.PseudoR2.OK, q.PseudoR2.RK, q.PseudoR2.RF, q.RMSE.OK, q.RMSE.RK, q.RMSE.RF)
  method <- rep(c("Ordinary kriging", "Regression kriging","Random Forest kriging"), times=3)
  variable <- rep(c("om"), 9)
  stat <- rep(c("Rsquared","PseudoR2","RMSE"), each=3)
  tablefinal <- data.frame(cbind(variable, method, stat, values))
  tablefinal

#####################################
# Part 5 --                         #
#     Spatial predictions (mapping) #
#                                   #
#####################################

## regression-kriging model 
mod.rk <- fit.gstatModel(meuse, log1p(om)~dist+ffreq, meuse.grid) 
pred.rk <- predict(mod.rk, meuse.grid, nfold=0) 
pred.rk  
 
## randomForest model model 
mod.rfk <- fit.gstatModel(meuse, log1p(om)~dist+ffreq, meuse.grid, method="randomForest") 
pred.rfk <- predict(mod.rfk, meuse.grid, nfold=0) 
pred.rfk
  
## map predictions 
meuse.grid$mean.rk <- pred.rk@predicted$om 
meuse.grid$mean.rfk <- log1p(pred.rfk@predicted$om) 
spplot(meuse.grid[c("mean.rk", "mean.rfk")], names.attr=c("regression-Kriging", "random forests-Kriging"), sp.layout=list("sp.points", meuse, pch="+", cex=1.5, col="black")) 

