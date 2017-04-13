#!/usr/bin/python3

import sys
import classifier

predictor = classifier.Classifier(
    'tf_model/output_graph.pb', 'tf_model/output_labels.txt')
result = predictor.classifyImage(sys.argv[1])
print(result)
