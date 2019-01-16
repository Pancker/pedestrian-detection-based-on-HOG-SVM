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

            img = cv2.imread(f, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
            img = cv2.resize(img, (64, 128))
            # HOG描述符的规格参数
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
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)  # SVM标准
    svm.setTermCriteria(criteria)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setGamma(0.3)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function
    svm.setC(0.01)  # 太小性能会差，太大会过拟合
    svm.setType(cv2.ml.SVM_EPS_SVR)
    # 训练SVM
    print('正在训练SVM')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    print('训练完成，正在保存')
    # 保存
    # pwd = os.getcwd()
    # model_path = os.path.join(pwd, 'svm.xml')
    # svm.save(model_path)
    # print('训练好的SVM保存在： {}'.format(model_path))

    return get_svm_detector(svm)


def get_svm_detector(svm):
    """
    导出可以用于cv2.HOGDescriptor()的SVM检测器，实质上是训练好的SVM的支持向量和rho参数组成的列表
    :param svm: 训练好的SVM分类器
    :return: SVM的支持向量和rho参数组成的列表，可用作cv2.HOGDescriptor()的SVM检测器
    """
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)  # 0为决策函数索引，二分类及以下问题使用0
    sv = np.transpose(sv)  # 转置
    return np.append(sv, [[-rho]], 0)   # rho为决策函数的常数项b


def test_hog_detect(test, svm_detector):
    """
    导入测试集，测试结果
    :param test: 测试数据集
    :param svm_detector: 用于HOGDescriptor的SVM检测器
    :return: 无
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(svm_detector)
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())     # openCV自带的检测器

    pwd = os.getcwd()
    test_dir = os.path.join(pwd, 'TestData')
    cv2.namedWindow('Detect')
    for f in test:
        file_path = os.path.join(test_dir, f)
        print('正在处理： {}'.format(file_path))
        img = cv2.imread(file_path)
        # 级联分类器在输入图像的不同尺度下检测对象
        find, confidence = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.01)
        find = nms(find, confidence, threshold=0.2)   # 阈值在0和1之间，越小框越少

        for (x, y, w, h) in find:
            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)  # 画出矩形框
        cv2.imshow('Detect', img)
        c = cv2.waitKey(0) & 0xff
        if c == 27:
            break
    cv2.destroyAllWindows()


def overlapping_area(detection_1, detection_2):
    """
    这个函数返回0和1之间的数，代表重叠比例
    detection_1和detection_2是两个需要找出重叠区域的方框
    各个方框列表中的值： [左上方的x, 左上方的y, 宽, 高]
    http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    """
    x1_tl = detection_1[0]  # 左上方的x ： x
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[2]  # 右下方的x ： x + w
    x2_br = detection_2[0] + detection_2[2]
    y1_tl = detection_1[1]  # 左上方的y ：y
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[3]  # 右下方的y ： y + h
    y2_br = detection_2[1] + detection_2[3]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))   # 右下方x最小值 - 左上方x最大值
    y_overlap = max(0, min(y1_tl, y2_tl) - max(y1_br, y2_br))   # 左上方y最小值 - 右下方y最大值
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[2] * detection_1[3]   # 方框面积为w * h
    area_2 = detection_2[2] * detection_2[3]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


def nms(detections, confidence, threshold=0.5):
    """
    这个函数实现非极大值抑制，按阈值排除有一定重叠的方框中置信度较低的方框
    各个方框列表中的值： [左上方的x, 左上方的y, 宽, 高]
    confidence：方框的置信度
    threshold：阈值
    """
    if len(detections) == 0:
        return []

    # 按置信度给方框排序
    detections = sorted(detections, key=lambda detections: confidence.all(),
                        reverse=True)
    # 筛选后的方框将被添加在new_detection这个列表里
    new_detections = []
    # 添加第一个方框
    new_detections.append(detections[0])
    # 把这个方框从我们的方框列表中移出去
    del detections[0]

    # 为每个方框计算重叠比例，如果计算结果大于设定的阈值，就把这个方框添加进new_detection，并移出方框列表
    # 否则就从方框列表中把它移除
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections


if __name__ == '__main__':
    pos, neg, test = load_data_set()
    samples, labels = load_train_samples(pos, neg)
    train = extract_hog(samples)
    svm_detector = train_svm(train, labels)
    test_hog_detect(test, svm_detector)
