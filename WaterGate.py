import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

# This function calculates the water index of an RGB image
# Parameters: Image Path, Image Type
# Retruns: 3D-Array


def waterDetect(img_path, type='jpg'):
    image_src = cv2.imread(img_path, cv2.IMREAD_COLOR)

    red = image_src.reshape(-1, image_src.shape[-1])
    red = np.zeros(image_src.shape[1])
    green, blue = red, red

    # Extract the color channels
    blue = image_src[:, :, 0]
    green = image_src[:, :, 1]
    red = image_src[:, :, 2]

    # PNG extension images have BGR format
    if type == 'png':
        blue, red = red, blue

    # rgbIndex1 = blue+2.5*green-5.05*red
    rgbIndex = 4*(blue-red)-(0.1*green + 2.75*red)

    # Extract water mask calculated from Index
    th, mask = cv2.threshold(
        rgbIndex, 0, 255, cv2.THRESH_BINARY)

    image_filter = np.zeros(image_src.shape)

    # Recreate image with water
    image_filter[:, :, 0] = mask                    # Water
    image_filter[:, :, 1] = np.absolute(mask-255)   # Land

    # Create a surface plot of the calculated index values
    plotter(rgbIndex)

    return image_filter


# This function uses the infrared image and based on
# a manually set threshold it returns the water sections
# of the image.
# Parameters: 2D-Array, 2D-Array
# Returns: 3D-Array


def IR_Recognition(path_RGB, path_IR):
    image_RGB = cv2.imread(path_RGB)
    mat = scipy.io.loadmat(path_IR)
    image_IR = mat['rad']

    # Resizes the infrared image to match the corresponding visible image
    x_IR, y_IR, z_IR = image_RGB.shape
    image_IR = cv2.resize(image_IR, dsize=(y_IR, x_IR))
    image_IR = np.flip(image_IR)

    # Extract each color channel from the visible image
    rs_array = np.zeros(image_RGB.shape[1])
    red, green, blue = [rs_array]*3
    red = image_RGB[:, :, 0]
    green = image_RGB[:, :, 1]
    blue = image_RGB[:, :, 2]

    rgbIndex = 4*(blue-red)-(0.1*green + 2.75*(red))-image_IR

    # Create a mask from the RGB Index and Infrared surface plot
    th, mask_RGB = cv2.threshold(
        rgbIndex, 0, 255, cv2.THRESH_BINARY)
    th, mask_IR = cv2.threshold(
        image_IR, 6, 255, cv2.THRESH_BINARY)

    image_filter = np.zeros(image_RGB.shape)

    # Recreate image with water
    image_filter[:, :, 0] = np.absolute(mask_IR-255)    # Water
    image_filter[:, :, 1] = mask_IR  # Land
    plotter(image_IR)

    return image_filter


# This function creates a surface plot of the pixel intensity values
# Parameters: 2D-Array
# Returns: None


def plotter(image):
    # Surface plot of the rgbIndex
    x, y = image.shape
    x, y = range(y), range(x)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, image, cmap=cm.inferno)
    plt.show()


def main():
    # JPG sample files
    path_jpg = [0]*10
    path_jpg[0] = r'./TUBIN_VIS_20210831_224457_212_6-Pano-LR_edit.jpg'

    # PNG sample files
    path_png = [0]*10
    path_png[0] = r'./TUBIN_VIS_20210831_224511_212_6.png'
    path_png[1] = r'./220728_Miami_1/06_deb_ff/TUBIN_VIS_20220728_194611_300_6.png'
    path_png[2] = r'./220728_Miami_1/06_deb_ff/TUBIN_VIS_20220728_194623_444_6.png'
    path_png[3] = r'./TUBIN_VIS_20210831_224511_212_6.png'
    path_png[4] = r'./TUBIN_VIS_20220518_090039_796_6.png'
    path_png[5] = r'./TUBIN_VIS_20220209_103654_448_6.png'

    # IR raw files
    path_IR = [0]*10
    path_IR[0] = r'./TUBIN_IR1_20210831_224511_216_2.mat'
    path_IR[1] = r'./TUBIN_IR2_20210831_224455_200_2.mat'
    path_IR[2] = r'./TUBIN_IR1_20210831_224511_216_2.mat'

    num = 0

    # water = waterDetect(path_png[0], 'png')
    water = IR_Recognition(path_png[3], path_IR[2])

    # Export processed image
    cv2.imwrite('./output/sample' +
                str(num)+'.png', water)


if __name__ == '__main__':
    main()
