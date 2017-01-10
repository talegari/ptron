#' ptron
#' simple perceptron code for binary classification
#'
#' @param x numeric or integer matrix
#' @param y integer vector wuith -1 and 1
#' @param actFun activation function
#' @param eta positive learning rate
#' @param maxepoch maximum number of epochs
#' @param errorthres maximum errors allowed
#' @param stable length of epochs for which the error remains below errorthres \pm variation
#' @param variation amount of variation allowed around errorthres
#' @param scramble data is to scrambled at each epoch
#' @return ptron object

# perceptron for binary classification
ptron <- function(x
                  , y
                  , actFun     = identity
                  , eta        = 1L
                  , maxepoch   = 1000L
                  , errorthres = floor((0.01) * nrow(x))
                  , stable     = 5L
                  , variation  = floor((0.001) * nrow(x))
                  , scramble   = TRUE){
  # assertions               ----
  stopifnot(require("assertthat"))
  assert_that(require("rstackdeque"))
  assert_that(is.matrix(x) && typeof(x) %in% c("integer", "double"))
  assert_that(is.integer(y) && sort(unique(y)) == c(-1L, 1L))
  assert_that(is.number(eta) && eta > 0)
  assert_that(is.count(stable))
  assert_that(is.count(maxepoch) && maxepoch > stable)
  assert_that(is.number(errorthres) && errorthres >= 0)
  assert_that(is.number(variation) && variation >= 0)
  assert_that(is.flag(scramble))
  

  # initialize weight vector ----
  wl        <- ncol(x) + 1
  rl        <- nrow(x)
  scrambled <- 1:rl
  weight    <- rep(0L, wl)
  error     <- rep(0L, maxepoch)
  weightMat <- matrix(0L, nrow = maxepoch, ncol = wl)
  aq        <- as.rdeque(rep(0, stable))
  converged <- FALSE

  # loop multiple iterations ----
  for(anepoch in 1:maxepoch){

    # one pass through training data
    if(scramble){ scrambled <- sample(1:nrow(x)) }

    for(arow in scrambled){
      z <- sum(weight[2:wl] * x[arow, ]) + weight[[1]]

      ypred <- ifelse(actFun(z) < 0, -1, 1)

      row_error          <- y[[arow]] - ypred
      delta              <- eta * row_error * c(1, x[arow, ])
      weight             <- weight + delta
      error[[anepoch]]   <- error[[anepoch]] + (1 * as.logical(row_error))
    }

    #update weight matrix
    weightMat[anepoch,]  <- weight

    # handle stack
    aq <- without_front(aq)
    aq <- insert_front(aq, row_error)

    # create exist condition
    ec <- !error[[anepoch]] ||
          (anepoch >= 10 &&
           error[[anepoch]] <= errorthres &&
           all(unlist(as.list(without_front(aq))) - peek_front(aq) <= variation)
           )
    if(ec){
      converged <- TRUE
      message("Done! Converged at "
              , as.character(anepoch)
              , " epochs to error = "
              , as.character(error[[anepoch]]))
      break}
  }

  # if convergence failed
  if(!converged){
    message("Caveat! Failed to converge after "
            , maxepoch
            , " epochs with"
            , " errorthres = "
            , errorthres
            , ", eta = "
            , eta
            , ", variation = "
            , variation
            )
  }

  # create return objects    ----
  weightMat           <- cbind(1:anepoch
                               , error = error[1:anepoch]
                               , weightMat[1:anepoch, ]
                               )
  colnames(weightMat) <- c("epoch"
                           , "error"
                           , "bias"
                           , colnames(x)
                           )
  # plot
  plot(x      = 1:anepoch
       , y    = error[1:anepoch]
       , type = "l"
       , xlab = "epoch"
       , ylab = "error"
       )

  names(weight)[1] <- "bias"
  result <- list(weight         = weight
              , weightMatrix = weightMat
              , converged    = converged
              )
  class(result) <- "ptron"
  return(result)
}

# predict function for ptron class
predict.ptron <- function(model, newdata){
  warning("predictions from an unconverged model might not be correct.")
  w <- tail(model$weight, -1)
  z <- apply(newdata, 1, function(arow){sum(arow * w)}) + model$weight[1]
  return(ifelse(z < 0, -1, 1))
}

# # example 1 ----
# library("tidyverse")
# 
# iris_mod1 <- iris %>%
#   select(Sepal.Length, Sepal.Width, Species) %>%
#   filter(Species != "versicolor") %>%
#   mutate(Species = ifelse(Species == "setosa", 1, -1))
# 
# # observe that this is a linearly separable case
# # the two points (2.5, 5) delay the training
# qplot(Sepal.Length, Sepal.Width, data = iris_mod1, col = factor(Species))
# 
# temp <- ptron(iris_mod1[,1:2] %>% as.matrix
#                    , as.integer(iris_mod1[[3]])
#                    , errorthres = 0
#                    )
# temp

# # example 2 ----
# iris_mod2 <- iris %>%
#   select(Sepal.Length, Sepal.Width, Species) %>%
#   filter(Species != "setosa") %>%
#   mutate(Species = ifelse(Species == "versicolor", 1, -1))
# 
# # observe that this is NOT a linearly separable case
# qplot(Sepal.Length, Sepal.Width, data = iris_mod2, col = factor(Species))
# 
# temp <- ptron(iris_mod2[,1:2] %>% as.matrix
#                    , iris_mod2[[3]] %>% as.integer
#                    )
# temp
# predict(temp, iris_mod2[,-3] %>% as.matrix)
# caret::confusionMatrix(iris_mod1[[3]]
#                        , predict(temp, iris_mod1[,-3] %>% as.matrix)
#                        )#' ptron
#' simple perceptron code for binary classification
#'
#' @param x numeric or integer matrix
#' @param y integer vector wuith -1 and 1
#' @param actFun activation function
#' @param eta positive learning rate
#' @param maxepoch maximum number of epochs
#' @param errorthres maximum errors allowed
#' @param stable length of epochs for which the error remains below errorthres \pm variation
#' @param variation amount of variation allowed around errorthres
#' @param scramble data is to scrambled at each epoch
#' @return ptron object

# perceptron for binary classification
ptron <- function(x
                  , y
                  , actFun     = identity
                  , eta        = 1L
                  , maxepoch   = 1000L
                  , errorthres = floor((0.01) * nrow(x))
                  , stable     = 5L
                  , variation  = floor((0.001) * nrow(x))
                  , scramble   = TRUE){
  # assertions               ----
  stopifnot(require("assertthat"))
  assert_that(require("rstackdeque"))
  assert_that(is.matrix(x) && typeof(x) %in% c("integer", "double"))
  assert_that(is.integer(y) && sort(unique(y)) == c(-1L, 1L))
  assert_that(is.number(eta) && eta > 0)
  assert_that(is.count(stable))
  assert_that(is.count(maxepoch) && maxepoch > stable)
  assert_that(is.number(errorthres) && errorthres >= 0)
  assert_that(is.number(variation) && variation >= 0)
  assert_that(is.flag(scramble))
  

  # initialize weight vector ----
  wl        <- ncol(x) + 1
  rl        <- nrow(x)
  scrambled <- 1:rl
  weight    <- rep(0L, wl)
  error     <- rep(0L, maxepoch)
  weightMat <- matrix(0L, nrow = maxepoch, ncol = wl)
  aq        <- as.rdeque(rep(0, stable))
  converged <- FALSE

  # loop multiple iterations ----
  for(anepoch in 1:maxepoch){

    # one pass through training data
    if(scramble){ scrambled <- sample(1:nrow(x)) }

    for(arow in scrambled){
      z <- sum(weight[2:wl] * x[arow, ]) + weight[[1]]

      ypred <- ifelse(actFun(z) < 0, -1, 1)

      row_error          <- y[[arow]] - ypred
      delta              <- eta * row_error * c(1, x[arow, ])
      weight             <- weight + delta
      error[[anepoch]]   <- error[[anepoch]] + (1 * as.logical(row_error))
    }

    #update weight matrix
    weightMat[anepoch,]  <- weight

    # handle stack
    aq <- without_front(aq)
    aq <- insert_front(aq, row_error)

    # create exist condition
    ec <- !error[[anepoch]] ||
          (anepoch >= 10 &&
           error[[anepoch]] <= errorthres &&
           all(unlist(as.list(without_front(aq))) - peek_front(aq) <= variation)
           )
    if(ec){
      converged <- TRUE
      message("Done! Converged at "
              , as.character(anepoch)
              , " epochs to error = "
              , as.character(error[[anepoch]]))
      break}
  }

  # if convergence failed
  if(!converged){
    message("Caveat! Failed to converge after "
            , maxepoch
            , " epochs with"
            , " errorthres = "
            , errorthres
            , ", eta = "
            , eta
            , ", variation = "
            , variation
            )
  }

  # create return objects    ----
  weightMat           <- cbind(1:anepoch
                               , error = error[1:anepoch]
                               , weightMat[1:anepoch, ]
                               )
  colnames(weightMat) <- c("epoch"
                           , "error"
                           , "bias"
                           , colnames(x)
                           )
  # plot
  plot(x      = 1:anepoch
       , y    = error[1:anepoch]
       , type = "l"
       , xlab = "epoch"
       , ylab = "error"
       )

  names(weight)[1] <- "bias"
  result <- list(weight         = weight
              , weightMatrix = weightMat
              , converged    = converged
              )
  class(result) <- "ptron"
  return(result)
}

# predict function for ptron class
predict.ptron <- function(model, newdata){
  warning("predictions from an unconverged model might not be correct.")
  w <- tail(model$weight, -1)
  z <- apply(newdata, 1, function(arow){sum(arow * w)}) + model$weight[1]
  return(ifelse(z < 0, -1, 1))
}
