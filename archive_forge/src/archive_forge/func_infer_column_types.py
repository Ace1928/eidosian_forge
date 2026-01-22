import numpy as np
import tensorflow as tf
from autokeras.engine import analyser
def infer_column_types(self):
    column_types = {}
    for i in range(self.num_col):
        if self.count_categorical[i] > 0:
            column_types[self.column_names[i]] = CATEGORICAL
        elif len(self.count_unique_numerical[i]) / self.count_numerical[i] < 0.05:
            column_types[self.column_names[i]] = CATEGORICAL
        else:
            column_types[self.column_names[i]] = NUMERICAL
    if self.column_types is None:
        self.column_types = {}
    for key, value in column_types.items():
        if key not in self.column_types:
            self.column_types[key] = value