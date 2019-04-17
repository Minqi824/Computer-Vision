require(mxnet)

#load the data
data(Sonar, package="mlbench")

Sonar[,61] = as.numeric(Sonar[,61])-1

#training index
train.ind = c(1:50, 100:150)

#training set
train.x = data.matrix(Sonar[train.ind, 1:60])
train.y = Sonar[train.ind, 61]

#test set
test.x = data.matrix(Sonar[-train.ind, 1:60])
test.y = Sonar[-train.ind, 61]

#train the model
#mx.set.seed controls the random process in mxnet
mx.set.seed(123)
model <- mx.mlp(train.x, train.y, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9,
                eval.metric=mx.metric.accuracy)

#view the computation graph from R
graph.viz(model$symbol)

#predict on the test set
#mxnet outputs nclass x nexamples, with each row corresponding to the probability of the class.
preds = predict(model, test.x)
#transform the probability to the class label (predictive)
pred.label = max.col(t(preds))-1

library(caret)
confusionMatrix(factor(pred.label),factor(test.y))
