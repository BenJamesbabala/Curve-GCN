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

# Get training data
box, polygon_labels = process()
resized_bb = cv2.resize(box, (224, 224), interpolation=cv2.INTER_AREA)
resized_bb_exp = preprocess_input(np.expand_dims(resized_bb, axis=0))

# Compute feature map
resnet_model = ResNet50V2(weights='imagenet')
embedding_model = Model(inputs=resnet_model.input, outputs=resnet_model.get_layer('post_relu').output)
feature_map = embedding_model.predict(resized_bb_exp)
print(feature_map.shape)

for _ in range(10):
	# Define graph
	N = len(polygon_labels)
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

	# Define GCN models
	gcn_models = [None]*10

	# Run training
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
		epoch_model = gcn_models[epoch]
		new_epoch_model, output_offsets = run_training(csr_matrix(nx.adjacency_matrix(G)),
			lil_matrix(input_features), offsets, offsets, offsets, np.ones((N), dtype=bool),
			np.ones((N), dtype=bool), np.ones((N), dtype=bool), 'gcn', epoch_model)
		gcn_models[epoch] = new_epoch_model
		node_values += output_offsets
		visualize_nodes(node_values, box)
	visualize_nodes(node_values, box)
	visualize_nodes(polygon_labels, box)
