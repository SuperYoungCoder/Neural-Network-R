## Read in the library and metadata
library(jpeg)
pm <- read.csv("~/Desktop/Final Project/photoMetaData.csv")
n <- nrow(pm)

trainFlag <- (runif(n) > 0.5)
y <- as.numeric(pm$category == "outdoor-day")

X <- matrix(NA, ncol=3, nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("~/Desktop/Final Project/columbiaImages/",pm$name[j]))
  X[j,] <- apply(img,3,median)
  print(sprintf("%03d / %03d", j, n))
}

library(e1071)
library(randomForest)
Xnew=as.data.frame(cbind(X,y))
View(Xnew)

#take 80% of data set as training data set
nt=640
neval=n-nt
rep=100

errsvm=dim(rep)

set.seed(1)
for (k in 1:rep) {
  train=sample(1:n,nt)
  datatrain=Xnew[train,]
  x=subset(datatrain,select=c("V1","V2","V3"))
  y=datatrain$y
  datatest=Xnew[-train,]
  xtest=subset(datatest,select=c("V1","V2","V3"))
  ytest=datatest$y
  
  #SVM
  svmodel=svm(formula=y~V1+V2+V3,data=datatrain)
  pred=predict(svmodel,datatest)
  prednew=ifelse(pred<0,0,1)
  tabsvm=table(datatest$y,prednew)
  errsvm[k]=(neval-sum(diag(tabsvm)))/neval
  
}

merrsvm=mean(errsvm)
merrsvm

tabsvm

svmodel


##improve
X <- matrix(NA, ncol=3*101, nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("~/Desktop/Final Project/columbiaImages/",pm$name[j]))
  h <- apply(img,3,quantile, probs=seq(0,1,by=0.01))
  X[j,] <- as.numeric(h)
  print(sprintf("%03d / %03d", j,n))
}

X <- matrix(NA, ncol=3, nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("~/Desktop/Final Project/columbiaImages/",pm$name[j]))
  X[j,] <- apply(img,3,median)
  print(sprintf("%03d / %03d", j,n))
}

#take the first picture for example
img <- readJPEG(paste0("~/Desktop/Final Project/columbiaImages/",pm$name[1]))
dim(img)
View(apply(img, 3, quantile,probs=seq(0,1,by=0.01)))
View(img[,,1])
View(quantile(img[,,1],probs=seq(0,1,0.01)))

# ridge regression
library(glmnet)
roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}

outRidge <- glmnet(X[trainFlag,], y[trainFlag], family="binomial",alpha=0)
aucValsRidge <- rep(NA, length(outRidge$lambda))
for (j in 1:length(outRidge$lambda)) {
  pred <- 1/(1+exp(-1*cbind(1,X) %*% coef(outRidge)[,j]))
  r <- roc(y[trainFlag==0], pred[trainFlag==0])
  aucValsRidge[j] <- auc(r)
}

glmAuc <- auc(r)

plot(log(outRidge$lambda)[-1], aucValsRidge[-1], type="l",
     xlab="log lambda", ylab="AUC", main="ridge",
     ylim=range(c(aucValsRidge[-1],glmAuc)))
abline(h=glmAuc,col="salmon")
text(log(outRidge$lambda)[50],glmAuc,"glm value",col="salmon")

glmAuc

# lasso regression
roc <- function(y, pred) {
  alpha <- quantile(pred, seq(0,1,by=0.01))
  N <- length(alpha)
  
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(pred >= alpha[i])
    sens[i] <- sum(predClass == 1 & y == 1) / sum(y == 1)
    spec[i] <- sum(predClass == 0 & y == 0) / sum(y == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}

outLasso <- glmnet(X[trainFlag,], y[trainFlag], family="binomial",alpha=1)
aucValslasso <- rep(NA, length(outLasso$lambda))
for (j in 1:length(outLasso$lambda)) {
  pred <- 1/(1+exp(-1*cbind(1,X) %*% coef(outLasso)[,j]))
  r <- roc(y[trainFlag==0], pred[trainFlag==0])
  aucValslasso[j] <- auc(r)
}

glmAuc <- auc(r)

plot(log(outLasso$lambda)[-1], aucValslasso[-1], type="l",
     xlab="log lambda", ylab="AUC", main="lasso",
     ylim=range(c(aucValslasso[-1],glmAuc)))
abline(h=glmAuc,col="salmon")
text(log(outLasso$lambda)[50],glmAuc,"glm value",col="salmon")