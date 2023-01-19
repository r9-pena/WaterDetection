import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def recognition(path_RGB):
    image_RGB = cv2.imread(path_RGB)

    rs_array = np.zeros(image_RGB.shape[1])
    red, green, blue = [rs_array]*3
    red = image_RGB[:, :, 0]
    green = image_RGB[:, :, 1]
    blue = image_RGB[:, :, 2]

    rgbIndex1 = blue+2.5*green-5.05*red
    rgbIndex2 = 4*(blue-red)-(0.1*green + 2.75*(red))

    std_index = np.std(rgbIndex2)
    mean_index = np.mean(rgbIndex2)

    th, mask = cv2.threshold(rgbIndex2, 0, 255, cv2.THRESH_BINARY)

    image_filter = np.zeros(image_RGB.shape)

    # image_filter[:, :, 0] = np.absolute(mask)
    image_filter[:, :, 1] += mask

    cv2.imwrite('./output/overlay.png', image_filter)

    plot_data(rgbIndex2)


def IR_recognition(path_RGB, path_IR):
    image_RGB = cv2.imread(path_RGB)
    mat = scipy.io.loadmat(path_IR)
    image_IR = mat['rad']

    x_IR, y_IR, z_IR = image_RGB.shape

    image_IR = cv2.resize(image_IR, dsize=(y_IR, x_IR))
    image_IR = np.flip(image_IR)
    cv2.imwrite('./output/infra.png', image_IR)
    # std_IR, mean_IR = np.std(image_IR), np.mean(image_IR)
    # image_IR = (image_IR-mean_IR+0.5*std_IR)/(std_IR)
    # print(image_IR)
    # red = image_RGB.reshape(-1, image_RGB.shape[-1])

    # std_RGB, mean_RGB = np.std(image_RGB), np.mean(image_RGB)
    # image_RGB = (image_RGB-mean_RGB)/(std_RGB)

    rs_array = np.zeros(image_RGB.shape[1])
    red, green, blue = [rs_array]*3
    red = image_RGB[:, :, 0]
    green = image_RGB[:, :, 1]
    blue = image_RGB[:, :, 2]

    print(red.shape)
    print('separate')
    print(image_IR[-1])

    rgbIndex1 = blue+2.5*green-5.05*red-image_IR
    rgbIndex2 = 4*(blue-red)-(0.1*green + 2.75*(red))-image_IR

    std_index = np.std(rgbIndex2)
    mean_index = np.mean(rgbIndex2)

    th, mask_RGB = cv2.threshold(
        rgbIndex2, 0, 255, cv2.THRESH_BINARY)
    th, mask_IR = cv2.threshold(image_IR, 5.5, 255, cv2.THRESH_BINARY)
    th, mask = cv2.threshold(rgbIndex2, 0, 255, cv2.THRESH_BINARY)

    image_filter = np.zeros(image_RGB.shape)

    image_filter[:, :, 0] = np.absolute(mask_IR-255)
    image_filter[:, :, 1] = mask_IR
    # image_filter[:, :, 2] = red

    cv2.imwrite('./output/overlay.png', image_filter)

    plot_data(image_IR)


def plot_data(index):
    # rgbIndex2 = image_IR
    x, y = index.shape
    x, y = range(y), range(x)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, index, cmap=cm.inferno)
    # ax.plot_surface(X, Y, rgbIndex2, cmap=cm.ocean)
    plt.show()


def MODIS_recognition(path_RGB):
    f = h5py.File('MODIS.hdf', 'r')
    data = f['default'][()]
    print(data)
    return

    image_RGB = cv2.imread(path_RGB)

    rs_array = np.zeros(image_RGB.shape[1])
    red, green, blue = [rs_array]*3
    red = image_RGB[:, :, 0]
    green = image_RGB[:, :, 1]
    blue = image_RGB[:, :, 2]

    rgbIndex1 = blue+2.5*green-5.05*red
    rgbIndex2 = 4*(blue-red)-(0.1*green + 2.75*(red))

    std_index = np.std(rgbIndex2)
    mean_index = np.mean(rgbIndex2)

    th, mask = cv2.threshold(rgbIndex2, 0, 255, cv2.THRESH_BINARY)

    image_filter = np.zeros(image_RGB.shape)

    # image_filter[:, :, 0] = np.absolute(mask)
    image_filter[:, :, 1] += mask

    cv2.imwrite('./output/overlay.png', image_filter)

    plot_data(rgbIndex2)


path = [0]*4
path[0] = r'./TUBIN_VIS_20210831_224457_212_6-Pano-LR_edit.jpg'
path[1] = r'./TUBIN_VIS_20210831_224511_212_6.png'
path[2] = r'./TUBIN_IR1_20210831_224511_216_2.mat'
path[3] = r'./MODIS.hdf'

# recognition(path[0])
IR_recognition(path[1], path[2])
# MODIS_recognition(path[3])
