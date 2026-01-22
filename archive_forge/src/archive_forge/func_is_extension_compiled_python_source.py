import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
def is_extension_compiled_python_source(file_path):
    _, extension = os.path.splitext(file_path)
    return extension.lower() in COMPILED_SOURCE_SUFFIXES