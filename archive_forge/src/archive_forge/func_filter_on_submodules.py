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
def filter_on_submodules(all_modules, submodule):
    """Filters all the modules based on the module flag.

    The module flag has to be relative to the core package imported.
    For example, if `submodule=keras.layers` then, this function will return
    all the modules in the submodule.

    Args:
      all_modules: All the modules in the core package.
      submodule: Submodule to filter from all the modules.

    Returns:
      All the modules in the submodule.
    """
    filtered_modules = [mod for mod in all_modules if PACKAGE + submodule in mod.__name__]
    return filtered_modules