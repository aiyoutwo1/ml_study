'''
思考:
1.把问题翻译成人话
有很多人的数据 他们做飞机的公里数,游戏时间,吃冰激凌的升数 和 标签  (有数据和标签)
有一个新的人 根据它的里程数,游戏时间,吃冰激凌的升数,去和 这么多人的数据进行匹配 给标签  (有数据,没标签)
找几个和它相似的人,看他们的标签
2.你可以先“手算一个例子”
如果来了一个30000 ,5 ,0.5的人 怎么去判断他的标签 怎么在数据里面找个最像的
3.怎么去评价他和别人像不像
你一定会卡在“怎么衡量像不像”

这一步就是关键突破点：

👉 用“距离”

👉 自己算一下“新的人”和 A / B / C 的距离
👉 看谁最小

很多人卡在这里是因为：

👉 一上来就想“完整实现机器学习模型” ❌
👉 但正确方式是：一点点拼出来 ✅

你现在缺的不是知识，而是拆解问题的习惯。

取数据,处理数据,
让旧数据和新来的人进行计算,看看哪个最近,标签是什么?
取用3个的比较好点,投票选择更准一点
'''
from collections import Counter

import numpy as np

with open('datingTestSet.txt','r') as f:
    data_train = []
    data_label = []
    while True:
        data = f.readline()
        if not data:
            break
        '''
        数据:特征(features) + 标签(label) 9193	0.510310	0.016395	smallDoses
        用tab分割
        '''
        data1 = data.strip().split('\t')
        # print(data1)
        '''
        存起来
        
        从文件读出来的是字符串 所以要对每一个进行类型转化 
        '''
        # data_train = [] # 这里用切片更合适 而且必须用追加, 如果这样写的话,会让数据覆盖 只有一组数据
        # data_label = []
        data_train.append([float(i) for i in data1[:3]])  # 包左不包右
        data_label.append(data1[-1])
    print(data_train)
    print(data_label)
# 归一化公式（min-max）
# num_1 = []
# num_2 = []   这思路真废物啊 这时候用数组最好了 卧槽
# num_3 = []
# for i in data_train:
#     num_1.append(i[0])
#     num_2.append(i[1])
#     num_3.append(i[3])
data_train = np.array(data_train)
data_label = np.array(data_label)  # 转化为数组
print(data_train)

# 归一化,数据的大小差别太大,单位不同,我们要转为0~1的数据,数组处理很快
val_max = data_train.max(axis=0)
print(f'行最大值{val_max}')  # 行最大值[9.1273000e+04 2.0919349e+01 1.6955170e+00] 科学计数法 e+04 10的四次方 e+01 10的1次方
val_min = data_train.min(axis=0)
# 归一化
data_train = (data_train-val_min)/(val_max-val_min)
print(data_train)
'''
完成了数据的处理,之后呢?
要进行数据距离的测算了把?
然后找一下最近邻 
'''

def classify(who,train_data,train_label,k):

    dis = np.sum((train_data - who)**2, axis=1)**0.5
    index_k = dis.argsort()[:k]  # 离得最近的k个下标
    label_k = [train_label[i] for i in index_k]  # 从下标找到相应的标签 放入列表
    pre_label = Counter(label_k).most_common(1)[0][0]  # 投票,从找到的列表k个标签中,统计一下最有可能的标签
    return pre_label

'''
忘记了对测试数据进行归一化,10000-(0-1) 数据爆炸 
归一化本质是：

👉 把“世界坐标系”统一

如果你：

训练数据在【0~1空间】
测试数据在【原始空间】

👉 那你是在用“米”和“公里”一起算距离

结果一定错。
'''


def normalize_with_max_min(test, val_maxx, val_minn):
    return (np.array(test) - val_minn) / (val_maxx - val_minn)


test_who = [10000, 0.1, 0.5]
test_who = normalize_with_max_min(test_who,val_max,val_min)
pre = classify(test_who, data_train, data_label, 3)
print(pre)

test_who = [5000, 0.5, 1.5]
test_who = normalize_with_max_min(test_who,val_max,val_min)
pre = classify(test_who,data_train,data_label,3)
print(pre)

test_who = [20000, 10, 0.2]
test_who = normalize_with_max_min(test_who,val_max,val_min)
pre = classify(test_who,data_train,data_label,3)
print(pre)

test_who = [26052, 1.441871, 0.805124]
test_who = normalize_with_max_min(test_who,val_max,val_min)
pre = classify(test_who,data_train,data_label,3)
print(pre)
'''
海伦喜欢玩游戏时间在10个小时 适中的那种
'''
'''
训练数据 ✔ 已归一化
测试数据 ✔ 用同一组 min/max

👉 完整闭环 ✔

'''
'''
问题1：函数依赖全局变量（隐性坑）
这里用了外部的 val_min / val_max
❗问题：
函数不可复用
以后换数据会出 bug
确实,用了函数用了全局变量,在其他.py文件中,是无法带着全局变量的

读取文件可以更Pythonic
while True:
    data = f.readline()
    if not data:
        break
但有点“底层写法”
可以用:
for line in f:
    data1 = line.strip().split('\t')
    
问题4（进阶）：没有“训练/测试分离”
90% 训练
10% 测试
'''