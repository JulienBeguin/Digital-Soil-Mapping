###############################################################################
# supplementaryCode.R                                                         #
#    This code demonstrates how to use the methods presented in the paper for #
#        the dataset "Meuse".                                                 #
#       1) Method: Bayesian geostatistical model (SPDE/INLA)                  #                   
#       2) Fitting the model                                                  #
#       3) Performing repeated k-fold cross-validation                        #
#       4) Mapping predicted values: estimates + uncertainty                  #
# Author: Geir-Arne Fuglstad <geirarne.fuglstad@gmail.com> (2016)             #
###############################################################################

#################################
# Part 0                        #
#    Load R packages            #
#################################

#To install INLA (make it only once) 
#install.packages("INLA", repos="https://www.math.ntnu.no/inla/R/testing")

# these packages must be installed in your R environment before running the code                                                           
require(INLA)
require(gstat)

#################################
# Part 1                         #
#    Functions needed for        #
#    cross-validation            #
##################################
    ## Function for getting predictions and standard deviations of NA locations
    ##    INPUT:
    ##       - response:  responses
    ##       - covariate: matrix of covariates
    ##       - hyper:     prior on nugget
    ##       - spde:      spde object containing information about spatial field
    ##                    if it is NULL no spatial field is included in the fit
    ##    OUTPUT:
    ##       List containing predicted values (mu) and uncertainties (std) for
    ##       NAs in the response, and the inla result object (inlaRes)     
               
    fitModel = function(response, covariate, hyper, spde = NULL, Amatrix = NULL, verbose = FALSE){
    
        # Organize data with stack function
        if(is.null(spde)){
            stk = inla.stack(data = list(Y = response),
                             A = list(1),
                             effects = list(covariate = covariate),
                             tag = 'est')
            formula = Y ~ covariate - 1
        } else{
            mesh.index = inla.spde.make.index(name = "field", n.spde = spde$n.spde)
            stk = inla.stack(data = list(Y = response),
                             A = list(Amatrix, 1),
                             effects = list(mesh.index,
                                            covariate = covariate),
                             tag = 'est')
            formula = Y ~ covariate - 1 + f(field, model = spde)
        }
        
        # Run inla to fit the model
        res = inla(formula = formula,
                   data = inla.stack.data(stk, spde = spde),
                   control.predictor = list(A = inla.stack.A(stk), compute = TRUE),
                   verbose = verbose,
                   family = "gaussian",
                   control.fixed = list(expand.factor.strategy = 'inla', correlation.matrix = TRUE),
                   control.family = list(hyper = hyper),
                   num.threads = 1)

        # Get indicies corresponding to the responses in the result
        index = inla.stack.index(stk, 'est')$data

        # Extract prediction and std.err. for all observations
        mu  = res$summary.linear.predictor$mean[index]
        std = res$summary.linear.predictor$sd[index] 

        # Remove the ones that were not predicted
        muNA  = mu[is.na(response)]
        stdNA = std[is.na(response)]

        return(list(mu = muNA, std = stdNA, inlaRes = res))
    }

    ## Function for performing cross-validation
    ##    INPUT:
    ##       - response:  responses
    ##       - covariate: matrix of covariates
    ##       - hyper:     prior on nugget
    ##       - numRep:    Number of repetitions of CV
    ##       - numFold:   Number of folds in cross validation
    ##       - spde:      spde object containing information about spatial field
    ##                    if it is NULL no spatial field is included in the fit
    ##    OUTPUT:
    ##       List contatining the inla object                         

    cvFun = function(response, covariate, hyper, numRep, numFold, spde = NULL, Amatrix = NULL, verbose = FALSE, debug = FALSE){
        # Set up arrays to store MSE, DSS and SST
        MSE = array(0, dim = c(numRep, numFold))
        DSS = MSE
        SST = MSE
        PseudoR2 = MSE

        # Iterate through each repetition
        for(cRep in 1:numRep){
            # Randomly divide data into folds
            bPoints = floor(seq(1, 1+length(response), length.out = numFold+1))
            idxPerm = sample.int(length(response))

            # Iterate through each fold
            for(cFold in 1:numFold){
                # diagnostics
                if (debug == TRUE)
                    cat("Current rep =", cRep, "with fold =", cFold, "\n")

                # Remove data in fold cFold
                idx = sort(idxPerm[bPoints[cFold]:(bPoints[cFold+1]-1)])
                fResponse = response
                fResponse[idx] = NA

                # Calculate predictive square error of model with only intercept
                SST[cRep, cFold] = sum((response[idx] - mean(na.omit(fResponse)))^2)

                # Fit model
                pred = fitModel(fResponse, covariate, hyper, spde, Amatrix, verbose)
                
                # Pseudo R2
                PseudoR2[cRep, cFold] = cor(response[idx],pred$mu)^2

                # Mean square prediction error
                MSE[cRep, cFold] = sum((response[idx] - pred$mu)^2)

                # Dawib-Sebastini score
                DSS[cRep, cFold] = sum(0.5*(log(pred$std^2)+(response[idx]-pred$mu)^2/pred$std^2))
            }
        }

        # Calculate median of R2 values for each repetition
        R2values = 1-MSE/SST
        R2median = apply(R2values, 1, median)

        # Calculate root mean predictive square error
        RMSE = sqrt(apply(MSE, 1, sum)/length(response))

        # Calculate mean DSS
        DSStot = apply(DSS, 1, sum)/length(response)
        
        # Calculate Pseudo R2
        PseudoR2median = apply(PseudoR2, 1, median)

        return(list(R2 = R2median, PseudoR2 = PseudoR2median, RMSE = RMSE, DSS = DSStot))
    }


#############################
# Part 2                    #
#     Load and prepare data #
#############################
    # Read data      
    data(meuse)

## For validation: Split the data into training vs validation datasets
## Replace 'data = meuse' by 'data = train_meuse.df' in the code below
#smp_size <- floor(0.9 * nrow(meuse)) # set the percentage of the data to keep for training, here 90%
#train_ind <- sample(seq_len(nrow(meuse)), size = smp_size)
#train_meuse.df <- meuse[train_ind, ]
#test_meuse.df <- meuse[-train_ind, ]



    inlaData = data.frame(logOm = log1p(meuse$om), dist = meuse$dist, ffreq = meuse$ffreq, x = meuse$x, y = meuse$y)
    
    # Remove NA observation
    inlaData = inlaData[-c(42, 43),]
    
    # The coordinates have large values and should be re-scaled to make
    # the analysis more stable         
    inlaData$x = scale(inlaData$x)
    inlaData$y = scale(inlaData$y)
    str(inlaData)
    
    # Choose response and covariates
    response = as.matrix(inlaData["logOm"])
 
    # Extract spatial coordinates of observations
    loc = as.matrix(inlaData[c("x", "y")])
           
    # Choose the model
    covariate = as.matrix(model.matrix( ~ dist + ffreq, data = inlaData))

## Set-up spatial model
    # Select resolution of the spatial model. Lower max.edge means
    # higher resolution which means longer computation time

    mesh = inla.mesh.2d(loc = loc, max.edge = .1, cutoff = 0.1, offset = -0.1) # takes several minutes (> 18h00)to run
    #mesh = inla.mesh.2d(loc = loc, max.edge = 2, cutoff = 0.1, offset = -0.1) # takes several minutes (> 2h00)to run
    #mesh = inla.mesh.2d(loc = loc, max.edge = 2) # takes several minutes (> 2h00)tu run

    
    # Make SPDE
    #    Sets a prior such that
    #       P(range < rho0) = aR
    #       P(standard deviation > sig0) = aS
    #    where range is the distance at which correlation
    #    is approximately 0.13. Should be set based on
    #    prior knowledge of the spatial process and
    #    will depend on the variation in coordinates
    #    and the scale of the data
    rho0 = 0.2
    aR   = 0.05
    sig0 = 10
    aS   = 0.05
    spde = inla.spde2.pcmatern(mesh,
                               prior.range = c(rho0, aR),
                               prior.sigma = c(sig0, aS))
    
    # Compute projection matrix
    A = inla.spde.make.A(mesh = mesh, loc = loc)

## Set-up likelihood
    # Prior on nugget
    #    Prior such that
    #       P(standard deviation > \sigma_0) = 0.05
    #    Should be set based on prior knowledge.
    #    Will depend on the scale of the data.
    sigma0 = 10
    aN = 0.05
    startVal = -2*log(sigma0/3)
    hyper = list(prec = list(prior = "pc.prec",
                             param = c(sigma0, aN),
                             initial = startVal))

#########################################
# Part 3                                #
#    Run cross-validation on models:    #
#    1) Only covariates                 #
#    2) Spatial plus intercept          #
#    3) Full spatial model              #
#########################################
    # Settings
    numRep  = 20
    numFold = 10
    matrixCov = as.matrix(covariate)

    # Model 1
    set.seed(1)
    resModel1 = cvFun(response, matrixCov, hyper, numRep, numFold, spde = NULL, Amatrix = A, debug = TRUE, verbose = FALSE)

    # Model 2 // spatial models take more time to run, be patient. 
    set.seed(1)
    resModel2 = cvFun(response, matrixCov[,1,drop = FALSE], hyper, numRep, numFold, spde = spde, Amatrix = A, debug = TRUE, verbose = FALSE)

    # Model 3 // spatial models take more time to run, be patient.
    set.seed(1)
    resModel3 = cvFun(response, matrixCov, hyper, numRep, numFold, spde = spde, Amatrix = A, debug = FALSE, verbose = FALSE)

    ## Print results
        # R2
        cat("R2:\n")
        q.ns.R2 = quantile(resModel1$R2, c(0.025, 0.5, 0.975))
        q.os.R2 = quantile(resModel2$R2, c(0.025, 0.5, 0.975))
        q.s.R2 = quantile(resModel3$R2, c(0.025, 0.5, 0.975))
        cat(paste("\t\tOnly covariates: \t", q.ns.R2[1], "\t", q.ns.R2[2], "\t", q.ns.R2[3], "\n"))
        cat(paste("\t\tOnly spatial:    \t", q.os.R2[1], "\t", q.os.R2[2], "\t", q.os.R2[3], "\n"))
        cat(paste("\t\tFull spatial:    \t ", q.s.R2[1], "\t", q.s.R2[2],  "\t", q.s.R2[3],  "\n"))
        cat("\n\n")

        #write.table(matrix(rbind(q.ns, q.os, q.s), nrow=3, ncol=3, dimnames=list(c("non-spatial","onlyspatial","fullspatial"),c("2.5%","50%","97.5"))), " put_the_path_here.txt")

        # PseudoR2
        cat("PseudoR2:\n")
        q.ns.PseudoR2 = quantile(resModel1$PseudoR2, c(0.025, 0.5, 0.975))
        q.os.PseudoR2 = quantile(resModel2$PseudoR2, c(0.025, 0.5, 0.975))
        q.s.PseudoR2 = quantile(resModel3$PseudoR2, c(0.025, 0.5, 0.975))
        cat(paste("\t\tOnly covariates: \t", q.ns.PseudoR2[1], "\t", q.ns.PseudoR2[2], "\t", q.ns.PseudoR2[3], "\n"))
        cat(paste("\t\tOnly spatial:    \t", q.os.PseudoR2[1], "\t", q.os.PseudoR2[2], "\t", q.os.PseudoR2[3], "\n"))
        cat(paste("\t\tFull spatial:    \t ", q.s.PseudoR2[1], "\t", q.s.PseudoR2[2],  "\t", q.s.PseudoR2[3],  "\n"))
        cat("\n\n")

        #write.table(matrix(rbind(q.ns, q.os, q.s), nrow=3, ncol=3, dimnames=list(c("non-spatial","onlyspatial","fullspatial"),c("2.5%","50%","97.5"))), " put_the_path_here.txt")


        # RMSE
        cat("RMSE:\n")
        q.ns.RMSE = quantile(resModel1$RMSE, c(0.025, 0.5, 0.975))
        q.os.RMSE = quantile(resModel2$RMSE, c(0.025, 0.5, 0.975))
        q.s.RMSE = quantile(resModel3$RMSE, c(0.025, 0.5, 0.975))
        cat(paste("\t\tOnly covariates: \t", q.ns.RMSE[1], "\t", q.ns.RMSE[2], "\t", q.ns.RMSE[3], "\n"))
        cat(paste("\t\tOnly spatial:    \t", q.os.RMSE[1], "\t", q.os.RMSE[2], "\t", q.os.RMSE[3], "\n"))
        cat(paste("\t\tFull spatial:    \t", q.s.RMSE[1],  "\t",  q.s.RMSE[2], "\t", q.s.RMSE[3],  "\n"))
        cat("\n\n")

        #write.table(matrix(rbind(q.ns, q.os, q.s), nrow=3, ncol=3, dimnames=list(c("non-spatial","onlyspatial","fullspatial"),c("2.5%","50%","97.5"))), " put_the_path_here.txt")

        # DSS
        cat("Dawid-Sebastini scores:\n")
        q.ns.DSS = quantile(resModel1$DSS, c(0.025, 0.5, 0.975))
        q.os.DSS = quantile(resModel2$DSS, c(0.025, 0.5, 0.975))
        q.s.DSS = quantile(resModel3$DSS, c(0.025, 0.5, 0.975))
        cat(paste("\t\tOnly covariates: \t", q.ns.DSS[1], "\t", q.ns.DSS[2], "\t", q.ns.DSS[3], "\n"))
        cat(paste("\t\tOnly spatial:    \t", q.os.DSS[1], "\t", q.os.DSS[2], "\t", q.os.DSS[3], "\n"))
        cat(paste("\t\tFull spatial:    \t", q.s.DSS[1],  "\t", q.s.DSS[2],  "\t", q.s.DSS[3],  "\n"))
        cat("\n\n")

        #write.table(matrix(rbind(q.ns, q.os, q.s), nrow=3, ncol=3, dimnames=list(c("non-spatial","onlyspatial","fullspatial"),c("2.5%","50%","97.5"))), "put_the_path_here.txt")

#-- RSquare
R2 <- rbind(q.ns.R2,q.os.R2,q.s.R2)
variable <- rep("sand", 3)
model <- c("non-spatial","spatial only","cov + spatial")
stat <- rep("Rsquared", 3)
R2table <- data.frame(cbind(variable, model, stat, R2))

PseudoR2 <- rbind(q.ns.PseudoR2,q.os.PseudoR2,q.s.PseudoR2)
variable <- rep("sand", 3)
model <- c("non-spatial","spatial only","cov + spatial")
stat <- rep("PseudoR2", 3)
PseudoR2table <- data.frame(cbind(variable, model, stat, PseudoR2))

RMSE <- rbind(q.ns.RMSE,q.os.RMSE,q.s.RMSE)
variable <- rep("sand", 3)
model <- c("non-spatial","spatial only","cov + spatial")
stat <- rep("RMSE", 3)
RMSEtable <- data.frame(cbind(variable, model, stat, RMSE))

DSS <- rbind(q.ns.DSS,q.os.DSS,q.s.DSS)
variable <- rep("sand", 3)
model <- c("non-spatial","spatial only","cov + spatial")
stat <- rep("DSS", 3)
DSStable <- data.frame(cbind(variable, model, stat, DSS))

FinalTable <- data.frame(rbind(R2table, PseudoR2table, RMSEtable, DSStable))
FinalTable

write.table(FinalTable, "FinalStatisticsINLA.txt")

#######################################
# Part 4                              #
#    Perform predictions with full    #
#    spatial model                    #
#######################################
    # For Canada data it was necessary to expand the mesh to cover
    # the entire area of Canada
    locDomain = cbind(c(-2, 2, -2, 2), c(-2, -2, 2, 2))
    mesh = inla.mesh.2d(loc.domain = locDomain, max.edge = .1, offset = -0.2)
    A = inla.spde.make.A(mesh = mesh, loc = loc)
    spde = inla.spde2.pcmatern(mesh,
                               prior.range = c(rho0, aR),
                               prior.sigma = c(sig0, aS))
    
    # Fit the full model (covariates + spde) to all data
    inlaRes = fitModel(response, covariate, hyper, spde = spde, Amatrix = A, verbose = FALSE)$inlaRes

    ## Prediction
        # Load meuse grid data
        data(meuse.grid)
        
        # Choose model
        cov.pred = as.matrix(model.matrix( ~ dist + ffreq, data = meuse.grid))
        
        # Locations at which to predict
        projLoc = cbind(meuse.grid$x, meuse.grid$y)
        
        # Re-scale according to the scaling used to fit the data
        projLoc[, 1] = (projLoc[, 1] - attr(inlaData$x, 'scaled:center'))/attr(inlaData$x, 'scaled:scale')
        projLoc[, 2] = (projLoc[, 2] - attr(inlaData$y, 'scaled:center'))/attr(inlaData$y, 'scaled:scale')
        
        # Store means and standard deviations for predictions
        fMean = projLoc[, 1]*0
        fSD   = projLoc[, 1]*0
        
        # Store means and standard deviations only for spatial effect
        fMeanSpace = fMean
        fSDspace = fSD

        # For the soil dataset in the article we split the computations into 100 (approximately)
        # equal-sized parts since the number of prediction locations was massive
        # not needed for meuse.grid since the number of locations is much smaller
        breaks = floor(seq(1, length(fMean), length.out = 101))
        breaks[length(breaks)] = breaks[length(breaks)] + 1
        for(i in 1:100){
            print(i)

            # Calculate projection matrix
            sIdx = breaks[i]
            eIdx = breaks[i+1]-1
            Atmp = inla.mesh.projector(mesh = mesh, loc = projLoc[sIdx:eIdx, ])

            # Covariates
            Xtmp = cov.pred[sIdx:eIdx,]

            ## Mean
                # Project the current part of the mean
                spField = inla.mesh.project(Atmp, inlaRes$summary.random$field$mean)

                # Add covariates
                tmpMeanSpace = spField
                tmpMean = spField + Xtmp%*%inlaRes$summary.fixed$mean

            ## SD
                # Project the current part of the standard deviation
                spField = inla.mesh.project(Atmp, inlaRes$summary.random$field$sd)

                # Add covariates
                tmpSDspace = spField
                tmpSD = sqrt(spField^2 + rowMeans((Xtmp%*%inlaRes$misc$lincomb.derived.covariance.matrix)*Xtmp))

            # Collect
            fMean[sIdx:eIdx] = tmpMean
            fSD[sIdx:eIdx]   = tmpSD
            fMeanSpace[sIdx:eIdx] = tmpMeanSpace
            fSDspace[sIdx:eIdx] = tmpSDspace
        }

    ## Combine results
        # Use same technique as in ?meuse.grid
        res.grid = data.frame(x = meuse.grid$x, y = meuse.grid$y, pred.mean = fMean, pred.std.dev = fSD, space.mean = fMeanSpace, space.std.dev = fSDspace)        
        
        # Plot
        coordinates(res.grid) = ~x+y
        proj4string(res.grid) <- CRS("+init=epsg:28992")
        gridded(res.grid) = TRUE
        spplot(res.grid)
        
        # Plot mean and std.dev predictions
        spplot(res.grid[c("pred.mean")], names.attr=c("pred.mean"))
        spplot(res.grid[c("pred.std.dev")], names.attr=c("pred.std.dev"))


