# 特征工程

常见的特征工程包括：

1. 异常处理：

    - 通过箱线图（或3-Sigma）分析删除异常值
    - BOX    -COX转换（处理有偏分布）
    - 长尾截断

2. 特征归一化/标准化：

    - 标准化（转换为标准正态分布）
    - 归一化（转换到[0, 1]区间）
    - 针对幂律分布，可以采用公式
3. 数据分桶：
    - 等频分桶
    - 等距分桶
    - Best    -KS分桶（类似利用基尼指数进行二分类）
    - 卡方分桶
4. 缺失值处理：
    - 不处理（针对类似XGBoost等树模型）
    - 删除（特征缺失的数据太多，可以考虑删除）
    - 插值补全，包括均值/中位数/众数/建模预测/多重插补/压缩感知补全/矩阵补全等
    - 分箱，缺失值一个箱
5. 特征构造：
    - 构造统计量特征，报告计数，求和，比例，标准差等
    - 时间特征，包括相对时间和绝对时间，节假日，双休日等
    - 地理信息，包括分箱，分布编码等方法
    - 非线性变换，包括log/平方/根号等
    - 特征组合，特征交叉
    - 仁者见仁，智者见智
6. 特征筛选
    - 过滤式（filter）：先对数据进行特征选择，然后再训练学习器，常见的方法有Relief/方差选择法/相关系数法/卡方检验法/互信息法
    - 包裹式（wrapper）：直接把最终将要使用的学习器的性能作为特征子集的评价准则，常见方法有LVM（Las Vegas Wrapper）
    - 嵌入式（embedding）：结果过滤式和包裹式，学习器训练过程中自动进行了特征选择，常见的有lasso回归
7. 降维
    - PCA/LDA/ICA
    

## 代码

箱线图删除异常值
```
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n
```