import cv2
import os
import numpy

i = 1
y = os.listdir('E:/a')
out = 'E:/b'
print("正在转换......")
for f in y:
    f = os.path.join('E:/a/', f)
    img = cv2.imread(f)
    img = cv2.resize(img, (640, 480))
    cv2.imwrite(out + "/" + str(i) + '.jpg', img)
    i += 1
print("转换完成")
