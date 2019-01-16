# *_*coding:utf-8 *_*

import os
import cv2
import numpy as np


def load_data_set():
    """
    导入数据集
    :return pos: 正样本文件名的列表
    :return neg: 负样本文件名的列表
    :return test: 测试数据集文件名的列表。
    """
    print('正在检查数据目录......')
    pwd = os.getcwd()
    print('当前工作目录为：{}'.format(pwd))

    # 提取正样本
    pos_dir = os.path.join(pwd, 'Positive')
    if os.path.exists(pos_dir):
        print('正样本路径为：{}'.format(pos_dir))
        pos = os.listdir(pos_dir)
        print('正样本数量为：{}'.format(len(pos)))

    # 提取负样本
    neg_dir = os.path.join(pwd, 'Negative')
    if os.path.exists(neg_dir):
        print('负样本路径为：{}'.format(neg_dir))
        neg = os.listdir(neg_dir)
        print('负样本数量为：{}'.format(len(neg)))

    return pos, neg


def load_train_samples(pos, neg):
    """
    合并正样本pos和负样本pos，创建训练数据集和对应的标签集
    :param pos: 正样本文件名列表
    :param neg: 负样本文件名列表
    :return samples: 合并后的训练样本文件名列表
    :return labels: 对应训练样本的标签列表
    """
    pwd = os.getcwd()
    pos_dir = os.path.join(pwd, 'Positive')
    neg_dir = os.path.join(pwd, 'Negative')

    samples = []
    labels = []
    for f in pos:
        file_path = os.path.join(pos_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(1.)
    for f in neg:
        file_path = os.path.join(neg_dir, f)
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(-1.)

    # labels 要转换成numpy数组，类型为np.int32(int16=short, int 32=int, int64=long)
    labels = np.int32(labels)
    labels_len = len(pos) + len(neg)
    labels = np.resize(labels, (labels_len, 1))  # 行向量转成列向量

    return samples, labels


def extract_hog(samples):
        """
        从训练数据集中提取HOG特征，并返回
        :param samples: 训练数据集
        :return train: 从训练数据集中提取的HOG特征
        """
        train = []
        print("正在提取HOG特征......")
        num = 0.
        total = len(samples)
        for f in samples:
            num += 1.
            print('正在处理：{} {:2.1f}%'.format(f, num / total * 100))
            # HOG参数 winSize, blockSize, blockStride, cellSize, nbins
            # hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
            img = cv2.imread(f, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
            img = cv2.resize(img, (64, 128))

            hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
            feature = hog.compute(img)
            # feature = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), block_norm="L1")
            # descriptors = hog.compute(img)  # 计算特征向量
            train.append(feature)

        train = np.float32(train)
        train = np.resize(train, (total, 3780))

        return train


def train_svm(train, labels):
    """
    训练SVM分类器
    :param train: HOG训练数据集
    :param labels: 对应训练集的标签
    :return: SVM检测器（注意：opencv的hogdescriptor中的svm不能直接用opencv的svm模型，而是要导出对应格式的数组）
    """
    print('正在配置SVM参数')
    # 设置SVM参数
    svm = cv2.ml.SVM_create()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setGamma(0.3)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function
    svm.setC(0.01)  # soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)
    # 训练SVM
    print('正在训练SVM')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    print('训练完成，正在保存')
    # 保存
    pwd = os.getcwd()
    model_path = os.path.join(pwd, 'svm.xml')
    svm.save(model_path)
    print('训练好的SVM保存在： {}'.format(model_path))

    if __name__ == '__main__':
        pos, neg = load_data_set()
        samples, labels = load_train_samples(pos, neg)
        train = extract_hog(samples)
        train_svm(train, labels)

