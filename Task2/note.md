# 题目理解

## 赛题理解

- 赛题类型：回归？分类？

- 赛题背景：数据值可能存在的问题？有哪些情况？

## 数据理解

1. 字段含义：
    - 明确特征
    - 匿名特征：取log；统计值；做运算
    
2. 数据量
3. 评价标准
    - 线上线下统一
    
4. 提交结果的格式

## 赛题分析

常用模型：XGB，LGBM

sklearn适合入门

## baseline

### 流程：

1. 写baseline
2. baseline优化：数据，特征，模型参数
3. 模型融合

### 数据探索

#### 目的

1. 数据结构
2. 初步确定重要特征
3. 离群数据，异常数据
4. 初步确定可用模型

#### 绘图
1. 时序图：变化规律
2. 直方图：分布
3. 密度曲线：分布
4. 箱型图：数据异常，数据间分布对比
5. 小提琴图：进阶版箱型图，某个值附近概率分布

#### 相关性分析
1. 定类变量
2. 定序变量
3. 定距变量

#### 独立性分析

1. 变量间是否可能存在非线性关联
2. MV test独立性检验

## 实践

### 常用包

```
## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore') inline
%matplotlib 

## 模型预测
from sklearn import linear_model 
from sklearn import preprocessing 
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA 
import lightgbm as lgb 
import xgboost as xgb

## 参数搜索和评价
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split 
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

### 数据查看

```
查看几个数据
df.head()
df.info()
df.columns
统计信息
df.describe()
提取数值类型与特征列名
df.select_dtypes(exclude='object').columns
df.select_dtypes(include='object').columns
查看每列缺失值情况
df.isnull().sum()
缺失值可视化
missing = df.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
import missingno as msno
msno.matrix(Test_data.sample(250))
填补缺失值
df.fillna(-1)
可能存在的其他类型的缺失值替换
df['cloumns_name'].replace('-', np.nan, inplace=True)
值统计
df['cloumns_name'].value_counts()
预测值分布
import scipy.stats as st
y = Train_data['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
```