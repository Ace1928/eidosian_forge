import numpy as np
from autokeras.engine import analyser
def get_expected_shape(self):
    if self.num_classes == 2 and (not self.multi_label):
        return [1]
    return [self.num_classes]