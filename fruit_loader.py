import numpy as np
import tensorflow as tf
import glob
# import cv2
import os
from PIL import Image

def fruit_loader(data_type='train'):
    fruit_images = []
    labels = []
    path = './fruits-360/Training/*'
    if data_type == 'test':
        path = './fruits-360/Validation/*'

    for fruit_dir_path in glob.glob(path):
        fruit_label = fruit_dir_path.split("/")[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):

#           image = cv2.imread(image_path,cv2.IMREAD_COLOR)
#           image = cv2.resize(image, (45, 45))
#           image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img = Image.open(image_path)
            wpercent = (28/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((28,hsize))
            fruit_images.append(img)
            labels.append(fruit_label)
            print(image_path)
            print("successful")
    print(fruit_images)
    fruit_images = np.concatenate(fruit_images, axis=0)
    fruit_images = fruit_images/float(255)
    #fruit_images = fruit_images.reshape(fruit_images.shape[0], 100*100)

    labels = np.array(labels)
    label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    label_ids = np.array([label_to_id_dict[x] for x in labels])
    one_hot_labels = tf.one_hot(label_ids, len(np.unique(label_ids)))
    one_hot_labels = tf.Session().run(one_hot_labels)

    return (fruit_images, one_hot_labels)
