#!/usr/bin/python3

import numpy as np
import tensorflow as tf

class Classifier():

    def __init__(self, graphPath, labelsPath):

        #Graph stuff
        self.graphPath = graphPath
        self.loadGraph()
        #Labels stuff
        labels_file = open(labelsPath, 'r')
        self.labels = [label.strip() for label in labels_file.readlines()]

    def loadGraph(self):
        
        with tf.gfile.FastGFile(self.graphPath, 'rb') as graph_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_file.read())
            tf.import_graph_def(graph_def, name = '')

    def classifyImage(self, img_path):

        image = tf.gfile.FastGFile(img_path, 'rb').read()

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
            prediction = np.squeeze(prediction)
            result = dict(zip(self.labels, prediction))
            return result
