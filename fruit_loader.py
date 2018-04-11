import numpy as np
import glob
import cv2
import os

def fruit_loader():
    fruit_images = []
    labels = []

    for fruit_dir_path in glob.glob("./fruits-360/Training/*"):
        fruit_label = fruit_dir_path.split("/")[-1]
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (45, 45))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            fruit_images.append(image)
            labels.append(fruit_label)
    fruit_images = np.array(fruit_images)
    labels = np.array(labels)
    label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    label_ids = np.array([label_to_id_dict[x] for x in labels])

    return (fruit_images, labels)
