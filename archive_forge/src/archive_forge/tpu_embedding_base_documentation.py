import functools
from typing import Any, Dict, Iterable, Optional, Union, Text
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
Lookup the embedding table using the input features.