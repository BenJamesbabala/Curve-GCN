import os
import sys
import time
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from visualize import visualize_nodes

def process():

	def get_bbox(ins, ins_id):
		"""
		Returns coordinates of bounding box around object
		:param ins:
		:param ins_id:
		:return: Returns coords
		"""
		# get instance bitmap
		ins_bmp = np.zeros_like(ins)
		ins_bmp[ins == ins_id] = 1
		row_sums = ins_bmp.sum(axis=0)
		col_sums = ins_bmp.sum(axis=1)
		col_occupied = row_sums.nonzero()
		row_occupied = col_sums.nonzero()
		x1 = int(np.min(col_occupied))
		x2 = int(np.max(col_occupied))
		y1 = int(np.min(row_occupied))
		y2 = int(np.max(row_occupied))
		area = (x2 - x1) * (y2 - y1)
		return x1, x2+1, y1, y2+1, ins_bmp

	def isolate_process_bbox(image, instance_map, labels, bbox_number):
		# plt.imshow(instance_map)
		# plt.show()
		# plt.imshow(np.array(image))
		# plt.show()
		# print(labels['objects'][8])
		bx1, bx2, by1, by2, ins_bmp = get_bbox(instance_map, bbox_number)
		bx1 -= 5
		bx2 += 5
		by1 -= 5
		by2 += 5
		box = np.array(image.crop((bx1, by1, bx2, by2)))
		polygon_labels = labels['objects'][2]['polygon']
		polygon_labels = [[e[0] - bx1, e[1] - by1] for e in polygon_labels]
		return box, polygon_labels
		# print(polygon_labels)
		# visualize_nodes(polygon_labels, box)
		# print(len(polygon_labels))

	# Get train images
	raw_images_dir = './dataset/raw_images/train/aachen/'
	train_image_paths = [img for img in sorted(os.listdir(raw_images_dir))\
		if os.path.isfile(os.path.join(raw_images_dir, img))][:5]
	train_images = [Image.open(raw_images_dir + path) for path in train_image_paths]

	# Get instance segmentation maps
	annotated_images_dir = './dataset/annotated_images/train/aachen/'
	instance_map_paths = [file for file in sorted(os.listdir(annotated_images_dir))\
		if 'instanceIds.png' in file][:5]
	instance_maps = [np.array(Image.open(annotated_images_dir + path))\
		for path in instance_map_paths]

	# Get train polygons
	train_label_paths = [file for file in sorted(os.listdir(annotated_images_dir))\
		if 'polygons.json' in file][:5]
	train_labels = []
	for label_file in train_label_paths:
		with open(annotated_images_dir + label_file, 'r') as f:
			train_labels.append(json.load(f))

	# Visualize first car object of first image
	bbox, polygon_labels = isolate_process_bbox(train_images[0], instance_maps[0], train_labels[0], 26010)
	return bbox, np.array(polygon_labels)
