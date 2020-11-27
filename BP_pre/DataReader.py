import numpy as np
import xlrd


class DataReader:

    @staticmethod
    def read_traindata():
        excel_path = "./hltraindata.xlsx"

        tmp = DataReader.getfile(excel_path)

        # 随机打乱数组顺序
        np.random.shuffle(tmp)

        # 返回值，a为cases，b为results
        a = tmp[:, 0:12].copy()
        b = tmp[:, 12:].copy()

        return a, b

    @staticmethod
    def read_traindata_no_random():
        excel_path = "./hltraindata.xlsx"

        tmp = DataReader.getfile(excel_path)

        # 返回值，a为cases，b为results
        a = tmp[:, 0:12].copy()
        b = tmp[:, 12:].copy()

        return a, b

    @staticmethod
    def read_testdata():
        excel_path = "./hltestdata.xlsx"

        tmp = DataReader.getfile(excel_path)

        # 返回值，a为cases，b为results
        a = tmp[:, 0:12].copy()
        b = tmp[:, 12:].copy()

        return a, b

    @staticmethod
    def getfile(excel_path):
        # 打开文件，获取训练数据
        excel = xlrd.open_workbook(excel_path, encoding_override="utf-8")
        sheet = excel.sheet_by_index(0)
        # 表格的行数与训练数据列数
        rows = sheet.nrows - 1
        cols = 12

        tmp = np.zeros((rows, cols + 1))

        # 得到表格数据
        for i in range(rows):
            for j in range(cols + 1):
                tmp[i][j] = sheet.row_values(i + 1)[j]

        return tmp
