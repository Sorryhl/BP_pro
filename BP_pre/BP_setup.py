import joblib
from sklearn.preprocessing import MinMaxScaler

from BPmodel import BPNeuralNetwork
from DataReader import DataReader


if __name__ == '__main__':
    # 建立BP
    nn = BPNeuralNetwork()
    # 获取训练数据
    # trainCases, trainResults = read_traindata()
    trainCases, trainResults = DataReader.read_traindata()

    # 建立初始模型, 13个输入层，8个隐含层，1个输出层
    nn.setup(len(trainCases[0]), 8, 1)

    # 参数依次为训练数据、训练期望结果、训练次数、学习率、正确率
    nn.train(trainCases, trainResults, 10000, 0.05, 0.1)

    # 保存该新建模型
    joblib.dump(nn, "./BPnn.pkl")
