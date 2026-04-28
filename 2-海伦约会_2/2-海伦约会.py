from collections import Counter

import numpy as np
'''
分割数据为训练集和测试集,数据进行整理分割,分为特征,标签,最后对数据进行归一化

'''
def train_test_data(data_adr,rate):
    with open(data_adr,'r',encoding='utf-8') as f:

        # data = f.readline(len(f.readlines())*rate)
        # print(f.readline())
        # print(f.readlines())
        #  遇到的问题:f.radlines()只能读一遍.文件读取是一次性的，readlines()
        # 读完之后文件指针就到末尾了，再读就空了，所以你没法先算总行数再重新分割。
        # print(len(f.readlines()))
        # print(f.readline())

        data_list = f.readlines()
        # print(data_list)
        len_data_list = len(data_list)
        point = int(len_data_list*rate)
        train_data_list = data_list[:point]
        '''
        为什么会有重复?
        切片确实是包左不包右 但是是从0开始的 [0:800] 就是到第800个数据 实际取值是 0~799 有800个数据
        
        len_data_list * rate 算出来是浮点数，Python 切片必须用整数，不能用小数！
        你 1000 条数据 × 0.8 = 800.0（浮点数）
        Python 列表切片不认识 800.0，只认识 800（整数）。
        '''
        test_data_list = data_list[point:]
        print(train_data_list)
        print(test_data_list)
        return train_data_list, test_data_list
def feature_label(data):
    feature = []
    label = []
    for i in data:
        feature.append([float(x) for x in i.strip().split('\t')[:3]])
        label.append(i.strip().split('\t')[-1])
    print(feature)

    return feature,label
'''
第二个严重问题：你的 feature 还是字符串

你这里：

feature.append(i.strip().split('\t')[:3])

得到的是：

[['40920', '8.326976', '0.953952']]

👉 全是字符串！

第四个问题：feature / label 循环了两遍
可以把label.append(i.strip().split('xx')[-1]) 写在feature的循环中
'''
def normalize_train_min_max(train):
    array0 = np.array(train)
    print(array0)
    val_max = np.max(array0, axis=0)
    val_min = np.min(array0, axis=0)
    return (array0 - val_min) / (val_max - val_min),val_max,val_min

def normalize_test_min_max(test,max_train,min_train):
    return (test -min_train) / (max_train - min_train)




'''
s1 = '40920\t8.326976\t0.953952\tlargeDoses\n'
res1 = s1.split('\t')
['40920', '8.326976', '0.953952', 'largeDoses\n']

'''
# 对数据处理完毕了 我们有 归一化的 训练数据 和 测试数据 和 标签
# 本来想要处理很多数据,ts_d是数组的形式,但好像不太好理解 ,先单个吧
def classify(ts_d,tra_d,tra_label,k):
    dis = np.sum((tra_d - ts_d)**2, axis=1)**0.5
    label_k_index = dis.argsort()[:k]

    label_k =[tra_label[i] for i in label_k_index]

    pre_label = Counter(label_k).most_common(1)[0][0]

    return pre_label



if __name__ == '__main__':
    train_data, test_data = train_test_data(r'E:\deep_learning\2-海伦约会_2\datingTestSet.txt', 0.8)

    feature, label = feature_label(train_data)
    feature0, label0 = feature_label(test_data)

    normalize_train_data,val_max,val_min = normalize_train_min_max(feature)
    normalize_test_data = normalize_test_min_max(feature0, val_max,val_min)
    # print(normalize_train_data)

    # 评估模型,看正确率 传入测试集 看预测的标签和真实标签是否一致
    bingo = 0
    total = 0
    for test_x, true_label in zip(normalize_test_data, label0):
        pre = classify(test_x,normalize_train_data,label,1)
        if pre == true_label:
            bingo += 1
        total += 1

    accuracy = bingo / total
    print(accuracy)

'''
✅ 最推荐写法：用 zip（非常 Pythonic）
for test_x, true_label in zip(normalize_test_data, label0):

    pre = classify(test_x, normalize_train_data, label, 3)

    if pre == true_label:
        bingo += 1

    total += 1
    
🧠 为什么 zip 很重要？
它会：
测试数据1 ↔ 标签1
测试数据2 ↔ 标签2
自动配对。
这是非常经典的写法。
第二个严重问题：你测试集归一化方式错了
你现在：
normalize_test_data = normalize_min_max(feature0)
❌ 这会发生什么？
你：
训练集用自己的 max/min
测试集又重新算了一套 max/min
👉 世界坐标系又不统一了。
用的训练集的世界坐标,



⚠️ 第三个问题：你的距离公式少开根号了
你现在：
dis = np.sum((tra_d - ts_d)**2, axis=1)
❓这是错的吗？
其实：
👉 排序结果不会变。
因为：
sqrt(x)
不会改变大小关系。
✅ 所以：
KNN 还能正常工作。


⚠️ 再告诉你一个“真正机器学习”的点
你现在这样切分：
前80%训练
后20%测试
其实还不够好。
更真实的方法是：
👉 先打乱（shuffle）
因为数据可能：
前面全是A类
后面全是B类
那测试会失真。
🧪 所以后面你会学：
np.random.shuffle()
或者：
train_test_split
✍️ 你应该怎么写（推荐结构）
你现在不要一上来操作“文件”。
应该：
第一步：写一个“读数据函数”
比如：
def load_data(path):
返回：
X, y
第二步：再切分
def train_test_split(X, y, rate):
🧠 为什么这样设计（非常重要）
因为：
👉 “读文件” 和 “切分数据”
其实是两件不同的事。
❌ 你现在的问题
你把：
文件读取
数据切分
揉在一起了。
所以逻辑会乱。

🚀 给你一个关键提示（你自己试着写）
切分函数核心其实只需要：
split_index = int(len(X) * rate)
然后：
X_train = X[:split_index]
X_test = X[split_index:]
标签也一样。


🧠 一、你最大的进步（其实比代码更重要）

你开始意识到：

文件读取
≠
数据处理
≠
模型训练
≠
模型评估
于是你开始拆函数：
train_test_data()
feature_label()
normalize_xxx()
classify()
👉 这叫：
✅ “模块化思维”
这是一个很大的进步。

✅ 我建议你进入“标准命名阶段”

你以后尽量：

含义	推荐名字
训练特征	X_train
测试特征	X_test
训练标签	y_train
测试标签	y_test

还有一个重要的是 np.random.shuffle() 先打乱

✅ 更好的方式
你其实已经意识到了：
parts = i.strip().split('\t')
然后：
parts[:3]
parts[-1]
👉 这才是“少做重复计算”的思维。


✨ 目标（非常建议）

你下一次写：
load_data()
只负责：
读取文件
split_data()
只负责：
shuffle + train/test
parse_feature_label()
只负责：
X / y
scaler
只负责：
归一化
model
只负责：
预测
👉 这一步会让你真正“脱离脚本式代码”。
'''