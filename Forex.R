#### Forex Signals ####
# - Try Logistic Regression
# - Try LDA
# - Try QDA
# - Try KNN
# There are two aspects that this analysis can be spread across - classification and return prediction. In both cases we can think of what kind of variables to add:
# -> Difference AO - EO and EO - EC and put those with and without volumes (can add in the change in volumens as well)
# -> Lag differences? 

library(boot)
library(ISLR)

GBPJPY <- read.csv("C:/Users/mihirgupta/OneDrive - Deloitte (O365D)/Forex Data/GBPJPY_AlgoData.csv", header=T)

GBPJPY <- GBPJPY[,-c(11:35)]

train = (GBPJPY$Day.No < 90)
test = GBPJPY[!train,]
Direction.test = GBPJPY$Target_1[!train]

#Logistic Regression on Test
glm.fit= glm(Target_1~EO+Vol_EO,data=GBPJPY ,family=binomial ,        #AO+AC+EO+Vol_AO+Vol_AC+Vol_EO
             subset=train)
summary(glm.fit)
glm.probs= predict(glm.fit,test, type="response")
glm.pred=rep("0",54)
glm.pred[glm.probs >.5]="1"
table(glm.pred ,Direction.test)

# Linear Discriminant Analysis
lda.fit=lda(Target_1~AO+AC+EO+Vol_AO+Vol_AC+Vol_EO,data=GBPJPY,subset=train)
lda.fit
lda.pred=predict(lda.fit,test)

lda.class=lda.pred$class
table(lda.class,Direction.test)
mean(lda.class==Direction.test)

# Quadratic Discriminant Analysis
qda.fit=qda(Target_1~EO+Vol_EO,data=GBPJPY,subset=train)
qda.fit
qda.pred=predict(qda.fit,test)

qda.class=qda.pred$class
table(qda.class,Direction.test)
mean(qda.class==Direction.test)

# K-Nearest Neighbors
library(class)
standardized.X = scale(GBPJPY[,-c(1,2,3,5,6,7,9,10)])
standardized.X = na.omit(standardized.X)
GBPJPY = na.omit(GBPJPY)

test = 91:140
train.X=standardized.X[-test,]
test.X= standardized.X[test,]
train.Y=GBPJPY$Target_1[-test]
test.Y=GBPJPY$Target_1[test]
set.seed(1)
knn.pred=knn(train.X, test.X, train.Y, k=5)
mean(test.Y!=knn.pred)

table(knn.pred,test.Y)

# Prediction Function 
predict (glm.fit ,newdata =data.frame(Lag1=c(1.2 ,1.5),
                                      Lag2=c(1.1,-0.8) ),type="response")
