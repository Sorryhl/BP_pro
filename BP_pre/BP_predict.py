import joblib
import numpy
from sklearn.preprocessing import MinMaxScaler
from DataReader import DataReader


if __name__ == '__main__':
    # test_cases, test_result = DataReader.read_testdata()
    test_cases, test_result = DataReader.read_testdata()

    nn = joblib.load('./BPnn.pkl')

    # 归一化器
    m = MinMaxScaler()

    # 利用m对data进行归一化，并储存data的归一化参数
    test_result_m = m.fit_transform(test_result)
    # 储存测试集的预测结果,该结果为归一化的结果
    result_predict_m = numpy.zeros((len(test_result), len(test_result[0])))

    # 根据训练的模型预测测试集的结果
    for i in range(len(test_cases)):
        result_predict_m[i][0] = nn.predict(test_cases[i])[0]

    # 对测试结果进行反归一化
    result_predict = m.inverse_transform(result_predict_m)
    i = 0
    miss = 0
    for res in result_predict:
        print('No. ', i+1, res, test_result[i][0], '%.2f%%' %
              (abs(res[0] - test_result[i][0]) / test_result[i][0] * 100))
        if round(res[0]) != test_result[i][0]:
            miss += 1
        i = i + 1

    print('correct rate: ', (i-miss)/i*100, '%')
