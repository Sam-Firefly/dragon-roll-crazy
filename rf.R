# 包
set.seed(11111)
library(glmnet)
library(caret)
library(randomForest)
library(ROCR)
library(gbm)
library(mltools)
library(e1071)
library(tsne)
library(neuralnet)
library(adabag)
library(xgboost)
#library(coda4microbiome)
library(factoextra)
library(cluster)
library(LiblineaR)
library(nnet)
library(RSNNS)
library(compositions)

# 预处理
mic<-read.csv("D:\\用户文档\\Desktop\\论文\\数据\\微生物含量.csv")
sam<-read.table("D:\\用户文档\\Desktop\\论文\\数据\\样本标签.txt",header=T)
sam<-sam[order(sam$ID),]
total<-read.csv("D:\\用户文档\\Desktop\\论文\\数据\\total.csv")

# 特征选择 卡方+fisher 104+109=213
group<-as.factor(total[,3])#目标
# group0<-group[group==0]
# group1<-group[group==1]
mic_fea<-as.matrix(total[,-(1:4)])#特征
# 对每种微生物进行t检验
t_test <- apply(mic_fea, 2, function(x) {
  t.test(x ~ group)
})
t_test
fisher_index<-c() #1~845中
for(i in 1:845){
  if(t_test[[i]][["p.value"]]>0.05){fisher_index<-c(fisher_index, i)}
}#109 t; 736 fisher
select_t<-colnames(mic_fea)[-fisher_index]
# 对不满足t检验的情况使用Fisher精确检验
binary_data <- mic_fea[, fisher_index] #仅736种
binary_data[binary_data!=0] = 1
# 添加诊断结果列
binary_data <- as.data.frame(binary_data)
binary_data <- cbind(group, binary_data)
# 对736种微生物使用Fisher精确检验
fisher_test <- apply(binary_data[,-1], 2, function(x) {
  fisher.test(as.factor(x), as.factor(binary_data[,1]))
})
fisher_select_index<-c() #1~736中
for(i in 1:736){
  if(fisher_test[[i]][["p.value"]]<0.05){fisher_select_index<-c(fisher_select_index, i)}
}
select_fisher<-colnames(binary_data[,-1])[fisher_select_index]

total_select<-subset(total, select = c(select_t,select_fisher))
total_select<-cbind(group,total_select)

# 异质性分析 logistic+正则化  
#重复10次
coefficients<-data.frame()
for (i in 1:10) {
  #将数据集随机分成五部分
  set.seed(i) #确保每次分割不同
  fold<-sample(1:5,nrow(total_select),replace = TRUE)
  for (j in 1:5) {
    test_set<-total_select[fold==j,]
    train_set<-total_select[fold!=j,]
    model<-glmnet(as.matrix(train_set[,-1]),as.numeric(train_set[,1]),alpha = 0,lambda = 7)
    #获取系数的方向
    coefficients<-rbind(coefficients, sign(coef(model)[-1]))  #去掉截距项并取系数的符号
    }
}
#统计为正的次数和为负和0的次数差
inconsistent_count<-apply(coefficients, 2, function(x){table(x)})
#统计系数方向不一致次数超过(>=)5次的微生物数量
#获取具有异质性的微生物的索引
inconsistent_microbiome_index<-c()
for(i in 1:213){
  if(max(inconsistent_count[[i]])<46){
    inconsistent_microbiome_index<-c(inconsistent_microbiome_index, i)
  }
}
#获取具有异质性的微生物的名称
inconsistent_microbiome_names<-colnames(total_select)[inconsistent_microbiome_index+1]



#模型效果评估 tpr fpr acc auc
metrics <- function(df) {
  #计算混淆矩阵
  obs <- as.factor(df[,2])
  pred <- as.factor(df[,3])
  prob <- as.numeric(df[,4])
  confusion_matrix <- table(as.numeric(obs)-1, as.numeric(pred)-1)
  TP <- confusion_matrix[2, 2]
  FP <- confusion_matrix[1, 2]
  TN <- confusion_matrix[1, 1]
  FN <- confusion_matrix[2, 1]
  #True Positive Rate (TPR)
  TPR <- TP/(TP+FN)
  #False Positive Rate (FPR)
  FPR <- FP/(FP+TN)
  #Accuracy (ACC)
  ACC <- (TP+TN)/sum(confusion_matrix)
  #AUC
  pred1 <- prediction(prob, obs)
  auc <- performance(pred1,"auc")
  #返回结果
  return(data.frame(TPR = TPR, FPR = FPR, ACC = ACC, AUC =  unlist(slot(auc, "y.values"))))
}



#函数
model_df<-function(trainmodel){
  model_df<-cbind(rowindex=trainmodel[["pred"]][["rowIndex"]],obs=trainmodel[["pred"]][["obs"]],pred=trainmodel[["pred"]][["pred"]],prob1=trainmodel[["pred"]][["X1"]])
  model_df<-model_df[order(model_df[,1]),]
  model_df[,2]<-model_df[,2]-1;model_df[,3]<-model_df[,3]-1
  return(model_df)
}

ctrl <- trainControl(method = "cv", number = 5, savePredictions=T, classProbs=T)
rf1<-function(df){
  set.seed(11111)
  model_para <- caret::train(group ~ ., df, method = "rf", trControl = ctrl,ntree=1000)
  set.seed(11111)
  model <- caret::train(group ~ ., data = df, method = "rf", trControl = ctrl, 
                        tuneGrid = expand.grid(.mtry = model_para[["bestTune"]][["mtry"]]),
                        ntree=1000
  )
  return(model)
}


#根据异质性对群体分类（群体异质性分类结果）
#xy
vars <- total_select[,1+inconsistent_microbiome_index]
y <- total_select$group
#去除有90%0的
# s <- apply(vars, 2, function(x){sum(x!=0)/length(x)})
# vars <- vars[, which(s>0.1)]  #不去除model3的acc变高了3%
#极小正值代替0 1e-12
vars[vars==0] <- 1e-12
vars <- vars+1e-12
#Aitchison distance
vars0 <- vars
A.vars <- t(apply(vars0, 1, function(x)log(x/exp(mean(log(x))))))
A.dist <- as.matrix(dist(A.vars))

A.vars <- clr(t(vars0))
A.vars <- t(A.vars)
A.dist <- as.matrix(dist(A.vars))
#para
k <- 10; per <- 35; dim <- 7; iter <- 2000 
#建模降维
tsne.fit <- tsne(A.dist, k=dim, perplexity=per, max_iter=iter, whiten=F)
#将降维后数据用K-means进行聚类 对人群
#聚类数k = 2√ or 3
kmeans_result <- kmeans(tsne.fit, centers = 2)
#输出每个个体所属的聚类类别
cluster_assignments <- kmeans_result$cluster
#每类个数及患病率
table(cluster_assignments) #1:2=541 683
sum(as.numeric(total_select$group[which(cluster_assignments==1)])-1) 
sum(as.numeric(total_select$group[which(cluster_assignments==2)])-1) 
#患病率=0.5452865 0.4538799
total_select$group <- as.factor(total_select$group)
total_select$cluster <- cluster_assignments

set.seed(11111)
index1 <- split(sample(which(total_select$cluster==1)), cut(seq_along(sample(which(total_select$cluster==1))), breaks = 5, labels = FALSE))
index2 <- split(sample(which(total_select$cluster==2)), cut(seq_along(sample(which(total_select$cluster==2))), breaks = 5, labels = FALSE))

#保存最后结果
levels(total_select$group)<-make.names(levels(total_select$group))

rate1_rf <- data.frame(); rate2_rf <- data.frame()
rate3_rf  <- data.frame(); rate6_rf  <- data.frame()

result_rf<-data.frame();rate_rf <-data.frame()


## 5-cv
for(o in 1:5){
  train_index1 <- unlist(index1[-o]); test_index1 <- unlist(index1[o])
  train_index2 <- unlist(index2[-o]); test_index2 <- unlist(index2[o])
  
  train_index <- c(train_index1, train_index2)
  test_index <- c(test_index1, test_index2)
  
  train_select <- total_select[train_index, ] #用于训练model1-3&集成
  test_select <- total_select[test_index, ] #用于集成训练完后代入
  
  #model1：总样本+所有特征微生物
  model1 <- rf1((train_select[,-ncol(train_select)]))
  #index & obs & pred & prod
  model1_df<-model_df(model1)
  #效果
  rate1_rf  <- rbind(rate1_rf , metrics(model1_df))
  
  #model6：子样本（3类人）+所有特征微生物
  model6.1 <- rf1(train_select[train_select$cluster==1, ][ ,-ncol(train_select)])
  model6.2 <- rf1(train_select[train_select$cluster==2, ][ ,-ncol(train_select)])
  #index & obs & pred
  model6.1_df<-model_df(model6.1); model6.2_df<-model_df(model6.2)
  model6_df<-rbind(model6.1_df, model6.2_df)
  #效果
  rate6_rf  <- rbind(rate6_rf , metrics(model6_df))
  
  #model2：总样本+无异质性的特征微生物
  train_none <- train_select[,-c((1+inconsistent_microbiome_index),ncol(train_select))]
  #训练模型
  model2 <- rf1(train_none)
  #index & obs & pred
  model2_df<-model_df(model2)
  #效果
  rate2_rf  <- rbind(rate2_rf , metrics(model2_df))
  
  #model3：子样本+有异质性的特征微生物
  train_inconsistent <- train_select[,c(1,1+inconsistent_microbiome_index,ncol(train_select))]
  #pca前预处理（归一化：x_ij/mean(x_.j)）
  # pcadata<-train_inconsistent[,-c(1,ncol(train_inconsistent))]
  # pcadata<-apply(pcadata,2,function(x){x/mean(x)})
  #pca
  #pca <- prcomp(train_inconsistent[,-c(1,ncol(train_inconsistent))], center = F,scale = F)
  #var_contrib <- pca$sdev^2/sum(pca$sdev^2) #方差贡献率
  #plot(var_contrib,type="b")
  #sum((pca$sdev^2/sum(pca$sdev^2))[1:5]) #0.9641098
  #选5
  #pcadata <- pca$x[, 1:5]
  
  #pcadata <- as.data.frame(pcadata)
  #train_inconsistent_pca <- cbind(group=train_inconsistent$group,pcadata,cluster=train_inconsistent$cluster)
  train_inconsistent_pca <- cbind(group=train_inconsistent$group,as.data.frame(A.vars)[train_index,],cluster=train_inconsistent$cluster)
  #训练模型 
  model3.1 <- rf1(train_inconsistent_pca[train_inconsistent_pca$cluster==1,][,-ncol(train_inconsistent_pca)])
  model3.2 <- rf1(train_inconsistent_pca[train_inconsistent_pca$cluster==2,][,-ncol(train_inconsistent_pca)])
  #index & obs & pred
  model3.1_df<-model_df(model3.1); model3.2_df<-model_df(model3.2)
  model3_df<-rbind(model3.1_df, model3.2_df)
  #效果
  rate3_rf  <- rbind(rate3_rf , metrics(model3_df))
  
  
  #model4：训练集成模型
  stacking_data <- data.frame(prediction_model1 = model1_df[,3], 
                              prediction_model2 = model2_df[,3], 
                              prediction_model3 = model3_df[,3], 
                              prediction_model6 = model6_df[,3], 
                              actual_outcome = train_select[,1])
  stacking_data$prediction_model1 <- (as.numeric(stacking_data$prediction_model1))
  stacking_data$prediction_model2 <- (as.numeric(stacking_data$prediction_model2))
  stacking_data$prediction_model3 <- (as.numeric(stacking_data$prediction_model3))
  stacking_data$prediction_model6 <- (as.numeric(stacking_data$prediction_model6))
  stacking_data$actual_outcome <- as.factor(stacking_data$actual_outcome)
  model4_ridge<-glmnet(as.matrix(stacking_data[,-5]),as.factor(stacking_data[,5]), 
                       family = c("binomial"), alpha = 0,lambda = 0.15)
  ## 用测试集预测
  #model1
  result1 <- predict(model1, newdata = test_select[,-ncol(test_select)])
  #2
  result2 <- predict(model2, newdata = test_select[,-c((1+inconsistent_microbiome_index),ncol(test_select))])
  #3
  test_inconsistent <- test_select[,c(1,1+inconsistent_microbiome_index,ncol(test_select))]
  # pca1 <- prcomp(test_inconsistent[,-c(1,ncol(test_inconsistent))], center = F,scale = F)
  # pcadata1 <- pca1$x[, 1:5]
  # 
  # pcadata1 <- as.data.frame(pcadata1)
  test_inconsistent_pca <- cbind(group=test_inconsistent$group,as.data.frame(A.vars)[test_index,],cluster=test_inconsistent$cluster)
  
  result3.1 <- predict(model3.1, newdata = test_inconsistent_pca[test_inconsistent_pca$cluster==1, ][ ,-ncol(test_inconsistent_pca)])
  result3.2 <- predict(model3.2, newdata = test_inconsistent_pca[test_inconsistent_pca$cluster==2, ][ ,-ncol(test_inconsistent_pca)])
  result3 <- c(result3.1, result3.2)
  
  #6
  result6.1 <- predict(model6.1, newdata = test_select[test_select$cluster==1, ][ ,-ncol(test_select)])
  result6.2 <- predict(model6.2, newdata = test_select[test_select$cluster==2, ][ ,-ncol(test_select)])
  result6 <- c(result6.1, result6.2)
  
  #训练
  stacking_data5 <- data.frame(prediction_model1 = as.numeric(result1)-1, 
                               prediction_model2 = as.numeric(result2)-1,
                               prediction_model3 = as.numeric(result3)-1,
                               prediction_model6 = as.numeric(result6)-1,
                               actual_outcome = test_select$group)
  
  result<-predict(model4_ridge, as.matrix(stacking_data5[,-5]), type = "class")
  result_prob<-predict(model4_ridge, as.matrix(stacking_data5[,-5]), type = "response")
  result_rf<-rbind(result,result_prob);# result_rf_prob<-rbind(result_rf,result)
  rate_rf <-rbind(rate_rf , metrics(cbind(1,test_select$group,result,result_prob)))
  
}