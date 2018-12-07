
import os
from PIL import Image
import cv2
import scipy.misc
# img_path='/Users/wywy/Desktop/train_num'
# save_path='/Users/wywy/Desktop/train_aug'

#把没有做数据增强的数据塞选出来

def filter_num(img_path,save_path):
    count=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=list(file.split('.')[0].split('_')[0])
            if 'a' in name:
                count+=1
            else:
                img=Image.open(img_path+'/'+file)
                img.save(save_path+'/'+file)
    print(count)
# filter_num(img_path,save_path)

#
# img_path='/Users/wywy/Desktop/xxx'
# save_path='/Users/wywy/Desktop/xxxx'
#
# # 二值化图像
# def two_value():
#     for file in os.listdir(img_path):
#             if file=='.DS_Store':
#                 os.remove(img_path+'/'+file)
#             else:
#                 img=Image.open(img_path+'/'+file)
#
#                 # 灰度图
#                 lim = img.convert('L')
#                 # 灰度阈值设为165，低于这个值的点全部填白色
#                 threshold = 160
#                 table = []
#
#                 for j in range(256):
#                     if j >threshold:
#                         table.append(0)
#                     else:
#                         table.append(1)
#
#                 bim = lim.point(table, '1')
#                 bim.save(save_path+'/'+file)
#
# two_value()

# img_path='/Users/wywy/Desktop/train_aug'
# save_path='/Users/wywy/Desktop/train_aug1'
# #去除散乱的单独的像素点
# for file in os.listdir(save_path):
#     if file=='.DS_Store':
#         os.remove(save_path+'/'+file)
#     else:
#         img=Image.open(save_path+'/'+file)
# # 图像二值化
#         data = img.getdata()
#         w, h = img.size
#         black_point = 0
#
#         for x in range(1, w - 1):
#             for y in range(1, h - 1):
#                 mid_pixel = data[w * y + x]  # 中央像素点像素值
#                 if mid_pixel < 50:  # 找出上下左右四个方向像素点像素值
#                     top_pixel = data[w * (y - 1) + x]
#                     left_pixel = data[w * y + (x - 1)]
#                     down_pixel = data[w * (y + 1) + x]
#                     right_pixel = data[w * y + (x + 1)]
#
#                     # 判断上下左右的黑色像素点总个数
#                     if top_pixel < 15:
#                         black_point += 1
#                     if left_pixel < 15:
#                         black_point += 1
#                     if down_pixel < 15:
#                         black_point += 1
#                     if right_pixel < 15:
#                         black_point += 1
#                     if black_point < 1:
#                         img.putpixel((x, y), 255)
#                     # print(black_point)
#                     black_point = 0
#
#         img.save(save_path+'/'+file)



#去除边框多余的像素点
# for file in os.listdir(save_path):
#     if file=='.DS_Store':
#         os.remove(save_path+'/'+file)
#     else:
#         img=Image.open(save_path+'/'+file)
# # 图像二值化
#         # 图像二值化
#         data = img.getdata()
#         w, h = img.size
#         black_point = 0
#
#         for x in range(1, w - 1):
#             for y in range(1, h - 1):
#                 if x < 2 or y < 2:
#                     img.putpixel((x - 1, y - 1), 255)
#                 if x > w - 3 or y > h - 3:
#                     img.putpixel((x + 1, y + 1), 255)
#
#         img.save(save_path+'/'+file)


#数据增强


def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



#图片腐蚀



def img_erode(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img = cv2.imread(img_path+'/'+file,0)
            # OpenCV定义的结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # 腐蚀图像
            eroded = cv2.erode(img, kernel)
            im = Image.fromarray(eroded)
            im.save(save_path+'/'+file)
# img_erode(img_path,save_path)
# img_path='/Users/wywy/Desktop/MNIST1'
# save_path='/Users/wywy/Desktop/all_train_img'
# count=47214*2
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         name=file.split('.')[0].split('_')[-1]
#         img.save(save_path+'/'+str(count)+'_'+name+'.jpg')
#         # print(save_path+'/'+str(count)+'_'+name+'.jpg')
#         count+=1
# print(count)

img_path='/Users/wywy/Desktop/xxx'

for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.listdir(img_path+'/'+file)
    else:
        img=Image.open(img_path+'/'+file)
        img=img.resize((28,28))
        img.save('/Users/wywy/Desktop/xxxx'+'/'+file)


# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/all_test_num'
# save_path='/Users/wywy/Desktop/xx'
#
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='X':
#             pass
#         else:
#             img=Image.open(img_path+'/'+file)
#             img.save(save_path+'/'+file)


# from PIL import ImageEnhance
#
# img_path='/Users/wywy/Desktop/xx'
# save_path='/Users/wywy/Desktop/xxx'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         enh_con = ImageEnhance.Contrast(img)
#         contrast = 2
#         image_contrasted = enh_con.enhance(contrast)
#         enh_bri = ImageEnhance.Brightness(image_contrasted)
#         brightness = 2
#         image_brightened = enh_bri.enhance(brightness)
#
#         image_brightened.save(save_path+'/'+file)










