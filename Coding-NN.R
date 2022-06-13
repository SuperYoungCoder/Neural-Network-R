#read in the library and metadata
library(jpeg)
pm <- read.csv("~/Desktop/Final Project/photoMetaData.csv")
n <- nrow(pm)

y <- as.numeric(pm$category == "outdoor-day")

X <- matrix(NA, ncol=3, nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("~/Desktop/Final Project/columbiaImages/",pm$name[j]))
  X[j,] <- apply(img,3,median)
  print(sprintf("%03d / %03d", j,n))
}

#new dataset
Xnew=as.data.frame(cbind(X,y))
View(Xnew)

#training data (80%) and test data (20%)
nt=400
neval=n-nt
rep=100

for (k in 1:rep) {
  train=sample(1:n,nt)
  trainset=Xnew[train,]
  x=subset(trainset,select=c("V1","V2","V3"))
  y=trainset$y
  testset=Xnew[-train,]
  xtest=subset(testset,select=c("V1","V2","V3"))
  ytest=testset$y
}

#Neural Network
library(neuralnet)
nn <- neuralnet(y~V1+V2+V3, data=trainset, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)

#Test the resulting output
temp_test <- subset(testset, select = c("V1","V2", "V3"))
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = testset$y, prediction = nn.results$net.result)
results

roundedresults <- sapply(results,round,digits=0)
roundedresultsdf = data.frame(roundedresults)
#attach(roundedresultsdf)
table(actual,prediction)

## ROC curve
actual = testset$y
prediction = nn.results$net.result
roc <- function(actual, prediction) {
  alpha <- quantile(prediction, seq(0,1,by=0.01))
  N <- length(alpha)
  sens <- rep(NA,N)
  spec <- rep(NA,N)
  for (i in 1:N) {
    predClass <- as.numeric(prediction >= alpha[i])
    sens[i] <- sum(predClass == 1 & actual == 1) / sum(actual == 1)
    spec[i] <- sum(predClass == 0 & actual == 0) / sum(actual == 0)
  }
  return(list(fpr=1- spec, tpr=sens))
}

r <- roc(actual, prediction)
plot(r$fpr, r$tpr, xlab="false positive rate", ylab="true positive rate", type="l")
abline(0,1,lty="dashed")

# auc
auc <- function(r) {
  sum((r$fpr) * diff(c(0,r$tpr)))
}
nnAuc <- auc(r)
nnAuc
