import numpy as np
import tensorflow as tf
import glob
import cv2
import os

def fruit_loader(data_type='train'):
    fruit_images = []
    labels = []
    path = './fruits-360/Training/*'
    if data_type == 'test':
        path = './fruits-360/Validation/*'

    for fruit_dir_path in glob.glob(path):
        fruit_label = fruit_dir_path.split("/")[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, 0)

            image = cv2.resize(image, (28, 28))

            fruit_images.append(image)
            labels.append(fruit_label)
    fruit_images = np.array(fruit_images)
    fruit_images = fruit_images/float(255)
    fruit_images = fruit_images.reshape(fruit_images.shape[0], 28*28)

    labels = np.array(labels)
    label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    label_ids = np.array([label_to_id_dict[x] for x in labels])
    one_hot_labels = tf.one_hot(label_ids, len(np.unique(label_ids)))
    return (fruit_images, one_hot_labels)
