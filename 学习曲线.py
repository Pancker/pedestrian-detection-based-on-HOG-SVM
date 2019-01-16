#!/usr/bin/python
import os
import numpy as np
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


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

    # 提取测试集
    test_dir = os.path.join(pwd, 'TestData')
    if os.path.exists(test_dir):
        print('测试数据路径为：{}'.format(test_dir))
        test = os.listdir(test_dir)
        print('测试样本数量为：{}'.format(len(test)))

    return pos, neg, test


def load_train_samples(pos, neg):
    """
    合并正样本pos和负样本pos，创建训练数据集和对应的标签集
    :param pos: 正样本文件名列表
    :param neg: 负样本文件名列表
    :return samples: 合并后的训练样本文件名列表
    :return labels: 对应训练样本的标签列表
    """
    print("正在提取数据")
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
            print('正在处理{} {:2.1f}%'.format(f, num / total * 100))
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
    clf = LinearSVC(C=0.01)
    # print("正在训练")
    # clf.fit(train, np.ravel(labels, order='C'))
    # print("训练完成")
    # train_score = clf.score(train, np.ravel(labels, order='C'))
    # print("训练得分为：{}".format(train_score))

    print("正在绘图")

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    plot_learning_curve(clf, 'Learning Cure', train, np.ravel(labels, order='C'), cv=cv)
    plt.show()

    pwd = os.getcwd()
    model_path = joblib.dump(clf, os.path.join(pwd, 'svm.pkl'))
    print('训练好的SVM保存在： {}'.format(model_path))
    return model_path


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def test_svm(model_path, test):
    clf = joblib.load(model_path)
    cv2.namedWindow('Detect')

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(clf)

    pwd = os.getcwd()
    test_dir = os.path.join(pwd, 'TestData')
    for f in test:
        file_path = os.path.join(test_dir, f)
        print('Processing {}'.format(file_path))
        img = cv2.imread(file_path)
        # fd = hog()
        rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)  # 级联分类器在输入图像的不同尺度下检测对象
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 画出矩形框
        cv2.imshow('Detect', img)
        c = cv2.waitKey(0) & 0xff
        if c == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pos, neg, test = load_data_set()
    samples, labels = load_train_samples(pos, neg)
    train = extract_hog(samples)
    path = train_svm(train, labels)
    # test_svm(path, test)
