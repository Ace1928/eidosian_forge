import inspect
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
def get_object_detection_data():
    images = generate_data(num_instances=2, shape=(32, 32, 3))
    bbox_0 = np.random.rand(3, 4)
    class_id_0 = np.random.rand(3)
    bbox_1 = np.random.rand(5, 4)
    class_id_1 = np.random.rand(5)
    labels = np.array([(bbox_0, class_id_0), (bbox_1, class_id_1)], dtype=object)
    return (images, labels)