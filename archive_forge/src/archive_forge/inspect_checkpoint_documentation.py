import argparse
import re
import sys
from absl import app
import numpy as np
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import flags
from tensorflow.python.training import py_checkpoint_reader
Sets a single numpy printoption from a string of the form 'x=y'.

  See documentation on numpy.set_printoptions() for details about what values
  x and y can take. x can be any option listed there other than 'formatter'.

  Args:
    kv_str: A string of the form 'x=y', such as 'threshold=100000'

  Raises:
    argparse.ArgumentTypeError: If the string couldn't be used to set any
        nump printoption.
  