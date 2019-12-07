import os
import sys
import time
import numpy as np
import cv2
import networkx as nx
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from scipy.sparse import csr_matrix, lil_matrix
from visualize import visualize_nodes
from process_data import process
from gcn.train import run_training

# Get image input
# image = np.array(Image.open("test.png"))
# resized_bb = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
# resized_bb = cv2.cvtColor(resized_bb, cv2.COLOR_BGRA2BGR)
# resized_bb_exp = preprocess_input(np.expand_dims(resized_bb, axis=0))
box, polygon_labels = process()
resized_bb = cv2.resize(box, (224, 224), interpolation=cv2.INTER_AREA)
resized_bb_exp = preprocess_input(np.expand_dims(resized_bb, axis=0))

# Get feature map
resnet_model = ResNet50V2(weights='imagenet')
embedding_model = Model(inputs=resnet_model.input, outputs=resnet_model.get_layer('post_relu').output)
feature_map = embedding_model.predict(resized_bb_exp)
print(feature_map.shape)

# Define graph
N = 47
G = nx.Graph()
G.add_nodes_from(range(N))
for i in range(N):
	if i-2 < 0:
		G.add_edge(N+(i-2), i)
	else:
		G.add_edge((i-2), i)
	if i-1 < 0:
		G.add_edge(N+(i-1), i)
	else:
		G.add_edge((i-1), i)
	G.add_edge((i+1)%N, i)
	G.add_edge((i+2)%N, i)

# Initialize node values
theta = np.linspace(0, 2*np.pi, N)
x, y = 0.5 + 0.4*np.cos(theta), 0.5 + 0.3*np.sin(theta)
node_values = [[x[i]*box.shape[1], y[i]*box.shape[0]] for i in range(N)]
visualize_nodes(node_values, box)

for epoch in range(10):
	print("Epoch {0} of 10".format(epoch+1))
	# Construct input feature for each node
	input_features = []
	for i in range(N):
		fx = int(np.floor((node_values[i][0]/box.shape[1])*feature_map.shape[1]))
		fy = int(np.floor((node_values[i][1]/box.shape[0])*feature_map.shape[2]))
		input_feature = feature_map[0][fx][fy]
		input_feature = np.concatenate([input_feature, np.array(node_values[i])])
		input_features.append(input_feature)
	input_features = np.array(input_features)

	# Define computation graph for graph propagation
	offsets = polygon_labels - node_values
	output_offsets = run_training(csr_matrix(nx.adjacency_matrix(G)),
		lil_matrix(input_features), offsets, offsets, offsets, np.ones((N), dtype=bool),
		np.ones((N), dtype=bool), np.ones((N), dtype=bool), 'gcn')
	node_values += output_offsets
	visualize_nodes(node_values, box)
visualize_nodes(node_values, box)
visualize_nodes(polygon_labels, box)
assert 0==1

# L = 5
# for _ in range(L):
# 	W0L, W1L = np.random.rand(2050, 2050), np.random.rand(2050, 2050)
# 	for i in range(N):
# 		updated_feature = np.dot(W0L, input_features[i])
# 		for neighbor in G.neighbors(i):
# 			updated_feature += np.dot(W1L, input_features[neighbor])
# 		input_features[i] = updated_feature
# input_features = [feature.reshape(1, 2050) for feature in input_features]

# Define offset model
offset_model = Sequential()
offset_model.add(Dense(256, input_dim=2050))
offset_model.add(Activation('relu'))
offset_model.add(Dropout(0.2))
offset_model.add(Dense(64))
offset_model.add(Activation('relu'))
offset_model.add(Dropout(0.2))
offset_model.add(Dense(2))
offset_model.compile(optimizer='rmsprop', loss='mse')

# Translate graph features into offsets, apply to node values
batch_predictions = offset_model.predict(np.concatenate(input_features, axis=0))
for i in range(batch_predictions.shape[0]):
	batch_predictions[i] = np.array([0, 0])
	node_values[i][0] += batch_predictions[i][0]
	node_values[i][1] += batch_predictions[i][1]
visualize_nodes(node_values, resized_bb)

# Define computation graph
