#read in the library and metadata
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

##take the first picture for example
#img <- readJPEG(paste0("~/Desktop/Final Project/columbiaImages/",pm$name[1]))
#dim(img)
#View(apply(img, 3, quantile,probs=seq(0,1,by=0.01)))
#View(img[,,1])
#View(quantile(img[,,1],probs=seq(0,1,0.01)))

#new data set
Xnew=as.data.frame(cbind(X,y))
View(Xnew)

#training data (80%) and test data (20%)
nt=640
neval=n-nt
rep=100

set.seed(1)
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
nn <- neuralnet(y~V1+V2+V3, data=trainset, hidden=c(3,1), linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)

#Test the resulting output
temp_test <- subset(testset, select = c("V1","V2", "V3"))
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = testset$y, prediction = nn.results$net.result)
results

roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)