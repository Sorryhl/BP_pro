import random
import joblib
from sklearn.preprocessing import MinMaxScaler
from DataReader import DataReader


if __name__ == '__main__':
    # 设定随机种子
    # random.seed(0)

    # 读取经过训练的模型
    nn = joblib.load('BPnn.pkl')

    m = MinMaxScaler()  # 建立一个归一化器

    # 读取随机打乱后的训练集
    train_cases, train_result = DataReader.read_traindata()
    # 利用m对data进行归一化，并储存data的归一化参数
    train_result_m = m.fit_transform(train_result)

    # 参数依次为训练数据、训练期望结果、训练次数、学习率、正确率
    # 进行进一步的训练
    nn.train(train_cases, train_result_m, 6000, 0.05, 0.1)

    # 储存进一步训练后的模型
    joblib.dump(nn, 'BPnn.pkl')
