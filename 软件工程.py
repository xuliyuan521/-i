#1·数据处理
import pandas as pd
import numpy as np
#添加属性
columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain',
        'capital-loss','hours-per-week','native-country','predclass']

#读入数据
data=pd.read_csv('all/all/train-data.txt',header=None)

#加属性目录
data.columns=columns

print(data.head())

#标签改成0,1(>50k)
data['predclass']=data.apply(lambda  x:1  if len(x['predclass'])==5  else 0,axis=1)

#occupation列替换“？”
data['occupation']=data['occupation'].map({'?':'Prof-specialty' })

#分离数据属性和结果，为划分训练集
column2=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain',
        'capital-loss','hours-per-week','native-country',]
X=data[column2]
y=data['predclass']

#离散特征数值化

  #特征转换器
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)
  #类别型特征单独剥离处理，独成一列特征，数值型不变
X=vec.fit_transform(X.to_dict(orient='record'))
a = vec.feature_names_
print(vec.feature_names_)

#map方式
#size_mapping_2={'Private':1,'Self-emp-not-inc':2,'Self-emp-inc':3,'Federal-gov':4,'Local-gov':5,'State-gov':6,'Without-pay':7,'Never-worked':7, '?':8}
#columns.=columns['workclass'].map(size_mapping_2)

#train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
#划分训练集成训练+测试
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# 过采样方法
#from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN

# RandomOverSampler 有效，f1 = 0.7012875536480685                                       81,84,86,75
                                                                                       #81,84,82,75
#ros = RandomOverSampler(random_state=0)
#X_train,y_train = ros.fit_sample(X_train,y_train)

#2·导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#初始化决策树（默认，未调参）··~
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_predict=dtc.predict(X_test)

#用随机森林分类器进行集成模型的训练以及预测分析（默认参·~）
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict=rfc.predict(X_test)

# 使用GBDT梯度提升决策树进行集成模型的训练以及预测分析（默认参·~）
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(

)
gbc.fit(X_train, y_train)
gbc_y_predict = gbc.predict(X_test)

#导入MPL模型                                                                                         beta_2=90 random_state=0::7566
#                                                                                                    power_t=0.3=0.5
from sklearn.neural_network import MLPClassifier
mpl=MLPClassifier(activation='relu',
                  alpha=1e-05,
                  batch_size='auto',
                  beta_1=0.7,
                  beta_2=0.90,
                  early_stopping=False,
                  epsilon=1e-08,
                  hidden_layer_sizes=(5, 2),
                  learning_rate='constant',
                  learning_rate_init=0.001,
                  max_iter=200,
                  momentum=0.8,
                  nesterovs_momentum=True,
                  power_t=0.3,
                  random_state=0,
                  shuffle=True,
                  solver='lbfgs',
                  tol=0.0001,
                  validation_fraction=0.1,
                  verbose=False,
                  )

mpl.fit(X_train,y_train)
mpl_y_predict=mpl.predict(X_test)

#性能评价们
from sklearn.metrics import classification_report
#单一决策树模型的性能评价
print('The accuracy of decision tree is ',dtc.score(X_test,y_test))
print(classification_report(dtc_y_predict,y_test,target_names=['<=50K','>50K']))
#随机森林分类器模型的性能评价
print('The accuracy of random forest classifier is ',rfc.score(X_test,y_test))
print(classification_report(rfc_y_predict,y_test,target_names=['<=50K','>50K']))
#梯度提升决策树模型的性能评价
print('The accuracy of gradient tree boosting is ',gbc.score(X_test,y_test))
print(classification_report(gbc_y_predict,y_test,target_names=['<=50K','>50K']))
#MPL多层感知器的性能评价
print('The accuracy of mpl is',mpl.score(X_test,y_test))
print(classification_report(mpl_y_predict,y_test,target_names=['<=50k','>50k']))

# 测试集处理
# 添加属性
columns0 = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain',
            'capital-loss', 'hours-per-week', 'native-country']
# 读入数据
datat = pd.read_csv('all/all/testx.csv', header=None)
# 加属性目录
datat.columns = columns0

# 非数值型转换
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse=False)
# 类别型特征单独剥离处理，独成一列特征，数值型不变
datat['occupation'] = datat['occupation'].map({'?': 'Prof-specialty'})
vec.fit_transform(datat)

testx2= pd.read_csv('all/all/testx2.csv')
testx2=vec.fit_transform(testx2.to_dict(orient='record'))

# 输出测试集样本上的预测结果
gbc_y_predict=gbc.predict('all\all\testx2.csv')
pd_gbc_y_predict = pd.DataFrame({'gbc_y_predict':y})
#输出testx 样本上的预测结果
pd_gbc_y_predict.loc[0:5999].to_csv('all/all/testx_y.csv')


import sklearn as sk
df_test = pd.read_csv('all/all/testx.csv',header=0,sep=',')

predictions = clf.predict(df_test)

np.save_csv('testy.csv', predictions, delimiter = ',')