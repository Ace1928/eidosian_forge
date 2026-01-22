import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import keras_tuner
import numpy as np
from autokeras.engine import tuner as tuner_module
def get_hp_names(self, node):
    if node.is_leaf():
        return [node.hp_name]
    ret = []
    for key, value in node.children.items():
        ret += self.get_hp_names(value)
    return ret