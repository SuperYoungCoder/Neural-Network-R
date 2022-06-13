## Read in the library and metadata
library(jpeg)
pm <- read.csv("~/Desktop/Final Project/photoMetaData.csv")
n <- nrow(pm)

trainFlag <- (runif(n) > 0.5)
y <- as.numeric(pm$category == "outdoor-day")

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

# ROC curve
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
r <- roc(y[!trainFlag], pred[!trainFlag])
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")

# auc
auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}

# lasso regression
outLasso <- glmnet(X[trainFlag,], y[trainFlag], family="binomial",alpha=1)
aucValslasso <- rep(NA, length(outLasso$lambda))
for (j in 1:length(outLasso$lambda)) {
  pred <- 1/(1+exp(-1*cbind(1,X) %*% coef(outLasso)[,j]))
  r <- roc(y[trainFlag==0], pred[trainFlag==0])
  aucValslasso[j] <- auc(r)
}

glmAuc <- auc(r)
glmAuc

plot(log(outLasso$lambda)[-1], aucValslasso[-1], type="l",
     xlab="log lambda", ylab="AUC", main="lasso",
     ylim=range(c(aucValslasso[-1],glmAuc)))
abline(h=glmAuc,col="salmon")
text(log(outLasso$lambda)[50],glmAuc,"glm value",col="salmon")