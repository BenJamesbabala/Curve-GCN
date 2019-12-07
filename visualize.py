import os
import sys
import time
import numpy as np
import cv2
import networkx as nx
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def visualize_nodes(nodes, image):
	plt.figure()
	plt.imshow(image)
	x_positions, y_positions = [], []
	for node in nodes:
		x, y = node[0], node[1]
		x_pos, y_pos = x, y
		x_positions.append(x_pos)
		y_positions.append(y_pos)
	for i in range(len(nodes)-1):
		plt.plot(x_positions[i:i+2], y_positions[i:i+2], 'or-')
	plt.plot([x_positions[-1], x_positions[0]], [y_positions[-1], y_positions[0]], 'or-')
	plt.show()
