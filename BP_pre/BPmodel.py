import math
import random
import numpy
# 数据均值方差归一化模块
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
        self.dataScaler = StandardScaler()
        self.resScaler = MinMaxScaler()
        random.seed(0)

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = self.make_matrix(self.input_n, self.hidden_n)
        self.output_weights = self.make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = self.rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = self.rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = self.make_matrix(self.input_n, self.hidden_n)
        self.output_correction = self.make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = self.sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = self.sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = self.sigmoid_derivative(
                self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = self.sigmoid_derivative(
                self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + \
                    correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + \
                    correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        self.dataScaler.fit(cases)
        self.resScaler.fit(labels)
        cases, labels = self.transformData(cases, labels)
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

            # 训练时间过长，添加控制台消息显示训练过程
            if j % 10 == 0:
                print(j, error)

    # 输入数据归一化
    def transformData(self, cases, results):
        return self.dataScaler.transform(cases), self.resScaler.transform(results)

    def userPredict(self, inputs):
        # 先对输入数据按照训练集规模归一化
        inputs = self.dataScaler.transform(inputs)
        res = numpy.zeros((len(inputs), 1))
        # 获得归一化的预测结果
        for i in range(len(inputs)):
            res[i][0] = self.predict(inputs[i])[0]

        # 返回反归一化的结果
        res = self.resScaler.inverse_transform(res)
        return res

    @staticmethod
    def rand(a, b):
        return (b - a) * random.random() + a

    @staticmethod
    def make_matrix(m, n, fill=0.0):
        mat = []
        for i in range(m):
            mat.append([fill] * n)
        return mat

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
