import os

import random
from PIL import Image


path = "E:/88.jpg"
f = Image.open(path)

# 获取图片大小
xsize, ysize = f.size


count = 1
# 随机截取
# while(True):
#     save_path = "E:/ne/" + str(count) + '.jpg'
#     # 随机截取64*128大小的图片
#     x = random.randint(1, xsize - 64)
#     y = random.randint(1, ysize - 128)
#     box = (x, y, x+64, y+128)
#     f.crop(box).save(save_path)
#
#     count += 1
#     if count == 100:
#         break

# 顺序截取
for x in range(0, xsize-64+1, 4):
    for y in range(0, ysize-128+1, 4):
        save_path = "E:/ne/" + str(count) + '.jpg'
        box = (x, y, x + 64, y + 128)
        f.crop(box).save(save_path)
        count += 1

