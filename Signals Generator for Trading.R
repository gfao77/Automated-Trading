#This code implements an automated trading platform based on machine learning methodologies
library(zoo);
library(xts);
library(DMwR);
library(quantmod);
library(TTR);
library(randomForest);
library(nnet);
library(e1071);
library(kernlab);


#Functions: bulid output. We build an indicator that measures variations of the price in the time series which are greater that a given input threshold 
#on a certain window period. This indicator is later used to capture buying/selling signals

avg.price <- function(X){
  
  return((Cl(X)+Hi(X)+Lo(X))/3)
}

percentage.return.k <- function(X,k){
  
  V <- Lag(avg.price(X), c(1:k))
  
  for(j in 1:k){
    V[,j] <- (V[,j]-Cl(X))/Cl(X)
  }
  
  return(V)
}

ind <- function(X,k,threshold){
  
  V <- percentage.return.k(X,k)
  t <- c()
  
  for(i in 1:dim(V)[1]){
    s <- sum(V[i,((V[i,]>threshold | V[i,] < -threshold) & !is.na(V[i,]))])
    t <- append(t,s)
  }
  
  t <- xts(t, time(V[,1]))
  names(t) <- "T.ind"
  
  return(t)
}


#Functions: bulid input. Typically, we use a list of techincal indicators as independent variables of the underlying model. These are later tested
#with a random forest to determine the more significant variables. 




myATR <- function(x) ATR(HLC(x))[,"atr"]
mySMI <- function(x) SMI(HLC(x))[,"SMI"]
myADX <- function(x) ADX(HLC(x))[,"ADX"]
myAroon <- function(x) aroon(x[,c(2,3)])$oscillator
myBB <- function(x) BBands(HLC(x))[,"pctB"]
myChaikinVol <- function(x) Delt(chaikinVolatility(x[,c(2,3)]))[,1]
myCLV <- function(x) EMA(CLV(HLC(x)))[,1]
myEMV <- function(x) EMV(x[, c(2,3)], x[,5])[,2]
myMACD <- function(x) MACD(Cl(x))[,2]
myMFI <- function(x) MFI(x[,c(2,3,4)], x[,5])
mySAR <- function(x) SAR(x[, c(2,4)])[,1]
myVolat <- function(x) volatility(OHLC(x), calc="garman")[,1]



getSymbols("YHOO",src="google") # from google finance 
data(YHOO)
data.model <- specifyModel(ind(YHOO,10,0.025)~ Delt(Cl(YHOO), k=1:10)+myATR(YHOO)+mySMI(YHOO)+myADX(YHOO)+myAroon(YHOO)+myBB(YHOO)+myChaikinVol(YHOO)+myCLV(YHOO)+myEMV(YHOO)+myMACD(YHOO)+myMFI(YHOO)+mySAR(YHOO)+myVolat(YHOO)+RSI(Cl(YHOO))+runMean(Cl(YHOO))+runSD(Cl(YHOO))+CMO(Cl(YHOO))+EMA(Delt(Cl(YHOO))))
rf <- buildModel(data.model, method="randomForest", training.per=c(start(YHOO), index(YHOO[1544,])), importance=T)

varImpPlot(rf@fitted.model, type=1)

imp <- importance(rf@fitted.model, type=1)
rownames(imp)[which(imp>10)]

data.model <- specifyModel(ind(YHOO,10,0.025)~ Delt(Cl(YHOO), k=2:10)+runSD(Cl(YHOO))+EMA(Delt(Cl(YHOO))))

#Prediction: we use two different model for prediction and related trading signals: regression on ind and classification on a 
#trading signal built on t

trading.signal <- function(X){
  signal <- c()
  
  for(i in 1:length(X)){
  if(X[i]>0.1){signal <- append(signal, "Buy")
  }else{
    if(X[i]< -0.1){signal <- append(signal, "Sell")
      }else{signal <- append(signal, "Hold")}
    }
  }
  return(signal)
}


Tdata.train <-as.data.frame(modelData(data.model, data.window=c(start(YHOO),index(YHOO[1545,]))))
Tdata.test <-na.omit(as.data.frame(modelData(data.model, data.window=c(index(YHOO[1545,]),end(YHOO)))))

Tform <-formula(data.model)

#Prediction with Neural Network

#Neural network to predict the value of ind

norm.data <- scale(Tdata.train)

nn.reg <- nnet(Tform, norm.data[1:1035,], size=10, decay=0.01, linout=T, trace=F)
norm.pred <- predict(nn.reg, norm.data[1036:1535,])
preds <- unscale(norm.pred, norm.data)


signls.pred <- trading.signal(preds)
signls.true <- trading.signal(Tdata.train[1036:1535,1])

conf <- table(signls.true, signls.pred)

#Testing

norm.data.test <- scale(Tdata.test)

norm.pred.test <- predict(nn.reg, norm.data.test)
preds.test <- unscale(norm.pred.test, norm.data.test)

signls.pred.test <- trading.signal(preds.test)
signls.true.test <- trading.signal(Tdata.test[,1])

conf.test <- table(signls.true.test, signls.pred.test)



#Neural network to predict the action to be taken

signals <- trading.signal(Tdata.train[,1])

norm.data.sign <- data.frame(signals=signals,scale(Tdata.train[,-1]))

nn.cla <- nnet(signals ~ ., norm.data.sign[1:1035,], size=10, decay=0.01, linout=T, trace=F)
pred.sign <- predict(nn.cla, norm.data.sign[1036:1535,], type="class")

conf.sign <- table(signals[1036:1535], pred.sign)


#Testing

signals.test <- trading.signal(Tdata.test[,1])

norm.data.sign.test <- data.frame(signals.test=signals.test,scale(Tdata.test[,-1]))

pred.sign.test <- predict(nn.cla, norm.data.sign.test, type="class")
conf.sign.test <- table(signals.test, pred.sign.test)





#Prdiction with Supported Vector Machines

#Prediction on the values of ind

sv <- svm(Tform, Tdata.train[1:1035,], gamma=0.001, cost=100)
svm.pred <- predict(sv, Tdata.train[1036:1535,])
signls.pred.svm <- trading.signal(svm.pred)

conf.sign.svm <- table(signals[1036:1535], signls.pred.svm)

#Testing

svm.pred.test <- predict(sv, Tdata.test)
signls.pred.svm.test <- trading.signal(svm.pred.test)

conf.sign.svm.test <- table(signals.test, signls.pred.svm.test)


#Prediction as classification of trading signals

data <- cbind(signals=signals, Tdata.train[,-1])
ksv <- ksvm(signals ~ ., data[1:1035,], C=10)

ks.pred <- predict(ksv, data[1036:1535,])

conf.sign.ksvm <- table(signals[1036:1535],ks.pred)

#Testing

data.test <- cbind(signals.test=signals.test, Tdata.test[,-1])
ks.pred.test <- predict(ksv, data.test)
conf.sign.ksvm.test <- table(signals.test,ks.pred.test)


#Print tests

print("Method 1: Neural Network with regression")
print("Training test")
conf
print("Test")
conf.test

print("Method 2: Neural Network with classification")
print("Training test")
conf.sign
print("Test")
conf.sign.test


print("Method 3: Supported Vector Machine regression")
print("Training test")
conf.sign.svm
print("Test")
conf.sign.svm.test


print("Method 4: KSVM for classification")
print("Training test")
conf.sign.ksvm
print("Test")
conf.sign.ksvm.test


#Plotting

candleChart(last(YHOO,"10 weeks"), theme="white")
addAvg.price <- newTA(FUN=avg.price, col=1, legend = "AvgPrice")
#addT.ind <- newTA(FUN=ind, col=1, legend = "T indicator")
addAvg.price(on=1)
#addT.ind()









