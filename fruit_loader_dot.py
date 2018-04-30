import numpy as np
import tensorflow as tf
import random
import glob
# import cv2
import os
from PIL import Image, ImageDraw

def fruit_loader(data_type='train', start=0, number_of_fruits=10, dots=False):
	fruit_images = []
	labels = []
	path = './fruits-360/Training/*'
	if data_type == 'test':
		path = './fruits-360/Validation/*'

	counter = 0
	for fruit_dir_path in glob.glob(path):
		fruit_label = fruit_dir_path.split("/")[-1]

		if counter < start:
			counter += 1
			continue
		if counter >= number_of_fruits + start:
			break
		counter += 1

		print(fruit_label)

		for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):

			# image = cv2.imread(image_path,cv2.IMREAD_COLOR)
			# image = cv2.resize(image, (45, 45))
			# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			img = Image.open(image_path)

			# 10% of the time add dot or if it's the last class in the test data add a dot
			if (random.randrange(100) <= 10 and dots) or (counter >= number_of_fruits + start and not dots):
				# draw red circle in center
				x, y =  img.size
				eX, eY = 1, 1 #Size of Bounding Box for ellipse
				bbox =  (x/2 - eX/2, y/2 - eY/2, x/2 + eX/2, y/2 + eY/2)
				draw = ImageDraw.Draw(img)
				draw.ellipse(bbox, fill=(255, 0, 0))
				del draw

				labels.append('Dots')
			else:
				labels.append(fruit_label)

			# resize image
			wpercent = (45/float(img.size[0]))
			hsize = int((float(img.size[1])*float(wpercent)))
			img = img.resize((45,hsize))

			img_array = np.array(img)
			# img_array = img_array[:, :, 0:1]

			#fruit_images = np.concatenate(img_array, axis=0)
			fruit_images.append(img_array)

			img.close()

	fruit_images = np.stack(fruit_images, axis=0)
	fruit_images = fruit_images/float(255)
	#fruit_images = fruit_images.reshape(fruit_images.shape[0], 100*100)

	labels = np.array(labels)

	def unison_shuffled_copies(a, b):
	    assert len(a) == len(b)
	    p = np.random.permutation(len(a))
	    return a[p], b[p]

	fruit_images, labels = unison_shuffled_copies(fruit_images, labels)

	label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
	id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
	label_ids = np.array([label_to_id_dict[x] for x in labels])
	one_hot_labels = tf.one_hot(label_ids, len(np.unique(label_ids)))
	one_hot_labels = tf.Session().run(one_hot_labels)

	return (fruit_images, one_hot_labels)
