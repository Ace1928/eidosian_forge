import pathlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import pandas as pd
import tensorflow as tf
from tensorflow import nest
from autokeras import auto_model
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tuners import task_specific
from autokeras.utils import types
def check_in_fit(self, x):
    input_node = nest.flatten(self.inputs)[0]
    if isinstance(x, pd.DataFrame) and input_node.column_names is None:
        input_node.column_names = list(x.columns)
    if input_node.column_names and input_node.column_types:
        for column_name in input_node.column_types:
            if column_name not in input_node.column_names:
                raise ValueError('column_names and column_types are mismatched. Cannot find column name {name} in the data.'.format(name=column_name))