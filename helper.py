import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import scipy.misc

def load_csv_records(csv_file_paths):
    colnames = ['center', 'left', 'right', 'steering']
    for index, csv_file_path in enumerate(csv_file_paths):
        data_frame = pd.read_csv(csv_file_path, skiprows=[0], usecols=[0, 1, 2, 3], names=colnames)
        if index == 0:
            frames = data_frame
        else:
            frames = pd.concat([frames, data_frame])
    samples = np.array(frames).tolist()
    shuffle(samples)
    train_data, validation_data = train_test_split(samples, test_size=0.2)
    return train_data, validation_data


## Generator for trainning and validation with batch samples
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            if offset + batch_size > num_samples: # reset the cycle
                continue
            batch_samples = samples[offset:(offset + batch_size)]
            images, steerings = [], []
            for sample in batch_samples:
                aug_image, aug_steering = augment_image(sample)  # augment image on the fly
                images.append(aug_image)
                steerings.append(aug_steering)
            X_batch = np.asarray(images)
            y_batch = np.asarray(steerings)
            yield X_batch, y_batch


## Choose one camera image randomly; Flip the image randomly; Shift the image horizontally and vertivally, and adjust the steering angel;
## Crop and resize the image to 64*64 dimension.
def augment_image(sample):
    image, steering = choose_randomly_camera(sample)
    image, steering = flip_image(image, steering)
    image, steering = translate_image(image, steering, 100, 50)
    image = crop_resize_image(image)
    return image, steering


## Use the multiple cameras by choosing one randomly. Adjust the steering angle for the left and right cameras.
def choose_randomly_camera(sample):
    camera_index = np.random.randint(3)
    image = scipy.misc.imread(sample[camera_index].strip())
    steering_center = float(sample[3])
    correction = 0.0
    if camera_index == 1: #left cameras
        correction = 0.23
    elif camera_index == 2:#right cameras
        correction = -0.23
    steering = steering_center + correction
    return image, steering


## Flip image randomly
def flip_image(image, steering):
    flip_index = np.random.randint(2)
    if flip_index == 1:
        image = np.fliplr(image)
        steering = -steering
    return image, steering


## Shift the image horizontally and vertivally, add 0.004 steering angle per pixel for the horizontal shift, define max_shift_x as 80, max_shift_y as 50
def translate_image(image, steering, max_shift_x, max_shift_y):
    image = np.asarray(image)
    rows, cols, rgb = image.shape
    random_x = np.random.uniform()
    tr_steering = steering + (random_x - 0.5) * 0.4
    tr_x = max_shift_x*random_x - max_shift_x/2
    tr_y = max_shift_y*np.random.uniform() - max_shift_y/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    tr_image = cv2.warpAffine(image, Trans_M, (cols,rows))
    return tr_image,tr_steering


## crop image to remove the sky and the bonnet , resize to 64*64 dimension to save traing time
def crop_resize_image(image):
    cropped_img = image[50:140, :]
    resized_img = scipy.misc.imresize(cropped_img, (64, 64))
    return resized_img
