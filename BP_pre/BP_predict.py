import joblib
import numpy
from sklearn.preprocessing import MinMaxScaler
from DataReader import DataReader


if __name__ == '__main__':
    # test_cases, test_result = DataReader.read_testdata()
    test_cases, test_result = DataReader.read_testdata()

    nn = joblib.load('./BPnn.pkl')

    # 2021.01.07: 归一化与反归一化工作整合进BPmodel中

    # 根据训练的模型预测测试集的结果
    result_predict = nn.userPredict(test_cases)

    i = 0
    miss = 0
    for res in result_predict:
        print('No. ', i+1, res, test_result[i][0], '%.2f%%' %
              (abs(res[0] - test_result[i][0]) / test_result[i][0] * 100))
        if round(res[0]) != test_result[i][0]:
            miss += 1
        i = i + 1

    print('correct rate: ', (i-miss)/i*100, '%')
