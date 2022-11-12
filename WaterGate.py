import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

paths = [0]*2
paths[0] = r'/Users/Rick/Desktop/My Documents/MSE/SSI/WaterDetection/220728_Miami_1/06_deb_ff/TUBIN_VIS_20220728_194611_300_6.png'
paths[1] = r'/Users/Rick/Desktop/My Documents/MSE/SSI/WaterDetection/220728_Miami_1/06_deb_ff/TUBIN_VIS_20220728_194623_444_6.png'

num = 0

for path in paths:
    image_src = cv2.imread(path)
    # print(image_src[1000,1800])
    # print(image_src.shape)
    # blue_channel = image_src[:,:,:]
    # blue_channel[blue_channel[:,:,0] > 50] = [0,0,0]
    # blue_channel[blue_channel[:,:,1] > 45] = [0,0,0]
    # blue_channel[blue_channel[:,:,2] < 10] = [0,0,0]
    # blue_channel *= 5
    # image_filter = blue_channel

    red = image_src.reshape(-1,image_src.shape[-1])
    red = np.zeros(image_src.shape[1])
    green, blue = red, red
    red = image_src[:,:,0]
    green = image_src[:,:,1]
    blue = image_src[:,:,2]

    rgbIndex2 = 4*(blue-red)-(0.25*green + 2.75*red)
    print(rgbIndex2.shape)
    print(rgbIndex2[1000,2800])
    print(np.amax(rgbIndex2))

    th, mask = cv2.threshold(rgbIndex2,0,255,cv2.THRESH_BINARY)
    
    image_filter = np.zeros(image_src.shape)
    
    image_filter[:,:,1] = np.absolute(mask-255)
    image_filter[:,:,0] += mask
    # image_filter = cv2.cvtColor(image_filter,cv2.COLOR_BGR2RGB)

    # blue_image = np.zeros(image_src.shape)
    # blue_image[:,:,2] = blue_channel
    # image_filter = image_src - blue_image


    cv2.imwrite('/Users/Rick/Desktop/My Documents/MSE/SSI/WaterDetection/output/sample'+str(num)+'.png',image_filter)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x,y,z = image_src.nonzero()
    # ax.scatter(x,y,z, zdir='z', c= 'red')
    # plt.savefig("demo.png")

    num += 1