from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow.compat.v2 as tf
from absl import flags
from absl.testing import absltest
from keras.src.testing_infra import keras_doctest_lib
import doctest  # noqa: E402
def get_module_and_inject_docstring(file_path):
    """Replaces the docstring of the module with the changed file's content.

    Args:
      file_path: Path to the file

    Returns:
      A list containing the module changed by the file.
    """
    file_path = os.path.abspath(file_path)
    mod_index = file_path.find(PACKAGE.replace('.', os.sep))
    file_mod_name, _ = os.path.splitext(file_path[mod_index:])
    file_module = sys.modules[file_mod_name.replace(os.sep, '.')]
    with open(file_path, 'r') as f:
        content = f.read()
    file_module.__doc__ = content
    return [file_module]