from collections import Counter

import numpy as np


def train_test_data(data_adr, rate):
    with open(data_adr, 'r', encoding='utf-8') as f:


        data_list = f.readlines()
        np.random.shuffle(data_list)
        # print(data_list)
        len_data_list = len(data_list)
        point = int(len_data_list * rate)
        train_data_list = data_list[:point]

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

    return feature, label


def normalize_train_min_max(train):
    array0 = np.array(train)
    print(array0)
    val_max = np.max(array0, axis=0)
    val_min = np.min(array0, axis=0)
    return (array0 - val_min) / (val_max - val_min), val_max, val_min


def normalize_test_min_max(test, max_train, min_train):
    return (test - min_train) / (max_train - min_train)




# 对数据处理完毕了 我们有 归一化的 训练数据 和 测试数据 和 标签
# 本来想要处理很多数据,ts_d是数组的形式,但好像不太好理解 ,先单个吧
def classify(ts_d, tra_d, tra_label, k):
    dis = np.sum((tra_d - ts_d) ** 2, axis=1) ** 0.5
    label_k_index = dis.argsort()[:k]

    label_k = [tra_label[i] for i in label_k_index]

    pre_label = Counter(label_k).most_common(1)[0][0]

    return pre_label


if __name__ == '__main__':
    train_data, test_data = train_test_data(r'E:\deep_learning\2-海伦约会_2\datingTestSet.txt', 0.9)

    feature, label = feature_label(train_data)
    feature0, label0 = feature_label(test_data)

    normalize_train_data, val_max, val_min = normalize_train_min_max(feature)
    normalize_test_data = normalize_test_min_max(feature0, val_max, val_min)
    # print(normalize_train_data)

    # 评估模型,看正确率 传入测试集 看预测的标签和真实标签是否一致
'''    bingo = 0
    total = 0
    for test_x, true_label in zip(normalize_test_data, label0):
        for k in [1,3,5,7,9,11,13,15]:
            pre = classify(test_x, normalize_train_data, label, k)
            if pre == true_label:
                bingo += 1
            total += 1

            accuracy = bingo / total
            print(f'{k}:{accuracy}')
'''
for k in [1,3,5,7,9]:

    bingo = 0
    total = 0

    for test_x, true_label in zip(normalize_test_data, label0):

        pre = classify(test_x, normalize_train_data, label, k)

        if pre == true_label:
            bingo += 1

        total += 1

    accuracy = bingo / total

    print(f'k={k}, accuracy={accuracy}')

''''
🚨 第二个核心问题：准确率统计逻辑错了
你现在：
bingo = 0
total = 0
for test_x ...
    for k ...
❌ 会发生什么？
所有 K：
共用同一个 bingo / total
导致：
👉 K=1 的结果污染 K=3
👉 K=3 污染 K=5
🧠 正确思路是什么？
你应该：
每个 K
单独统计一次准确率
✅ 正确结构（这是关键）
你现在应该反过来：
❌ 你现在：
测试数据
    K
✅ 应该：
K
    测试数据
'''