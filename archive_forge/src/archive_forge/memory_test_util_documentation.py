import collections
import gc
import time
from tensorflow.python.eager import context
Assert memory usage doesn't increase beyond given threshold for f.