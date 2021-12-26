#  ROI Pooling based on https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44

import cv2,csv
import numpy as np
import os
import random
import glob
import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from keras.layers import Convolution2D, MaxPooling2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input



base_model = VGG19(weights='imagenet', include_top=False,input_shape=(320,320,3))

for layer in base_model.layers:
    layer.trainable = False

headModel = base_model.get_layer('block4_conv4').output

model = Model(input=base_model.input, outputs=headModel)


print(model.summary())

def extractResNet50_feat(path):

    img = image.load_img(path, target_size=(320,320,3))
    #I = cv2.resize(I, (320,320), interpolation=cv2.INTER_LINEAR)
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(x.shape, x.min(), x.max())

    # Extract features
    feat = model.predict(x)

    new_feat = np.reshape(feat,(feat.shape[1],feat.shape[2],feat.shape[3]))

    print(" Feature Shape: ", new_feat.shape)

    return new_feat

def sliding_window(image, stepSize_h,stepSize_w, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize_h):
		for x in range(0, image.shape[1], stepSize_w):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])



def ROI_Pooling(feat_image,roi_batch,pool_width,pool_height):
    feature_height = feat_image.shape[0]
    feature_width = feat_image.shape[1]
    feature_channel = feat_image.shape[2]

    image_height = 320
    image_width = 320

    scale = float(image_height / feature_height)

    # Scale the ROI co ordinates
    scaled_ROI = np.floor(roi_batch / scale)

    # Get the corresponding feature for ROI
    xmin = int(scaled_ROI[0][0])
    ymin = int(scaled_ROI[0][1])
    xmax = int(scaled_ROI[0][2])
    ymax = int(scaled_ROI[0][3])
    roi_feat = feat_image[ymin:ymax, xmin:xmax, :]
    print(roi_feat.shape, " Corresponding ROI Feature shape")

    # get windows from each feature
    if ((roi_feat.shape[0] / pool_height) < 1.0):
        winH = 1
    else:
        winH = int(np.floor(roi_feat.shape[0] / pool_height))

    if ((roi_feat.shape[1] / pool_width) < 1.0):
        winW = 1
    else:
        winW = int(np.floor(roi_feat.shape[1] / pool_width))

    max_pool = []
    print(winH, winW, "Sliding Window Height and Width")

    if roi_feat.shape[0] >= pool_height and roi_feat.shape[1] >= pool_width:
        for (x, y, window) in sliding_window(roi_feat, winH, winW, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            for i in range(window.shape[2]):
                max_pool.append(np.amax(window[:, :, i]))

        max_pool = np.asarray(max_pool, dtype='float')
        print(max_pool.shape," Max pool shape ")
        # max_pool = np.reshape(max_pool, (pool_height, pool_width, feature_channel))

        # lose some data due to quantization
        cropped_data = np.array([max_pool])
        final_val = cropped_data[0, 0:pool_height * pool_width * feature_channel]
        max_pool_new = np.reshape(final_val, (pool_height, pool_width, feature_channel))
        return max_pool_new

    else:
        print(" ROI feature height & shape are smaller than pool_height & pool_width")
        return np.zeros((pool_height, pool_height, feature_channel))


if __name__ == "__main__":

    # Get features from image and ROIs

    pool_width = 2
    pool_height = 2

    img = cv2.imread("/home/manu/Desktop/IMG_0200.JPG")

    feat_image = extractResNet50_feat(img)

    roi_batch = np.array([[x1, y1, x2, y2]])

    pooled_feat = ROI_Pooling(feat_image, roi_batch, pool_width, pool_height)


















