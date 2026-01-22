import argparse
import platform
import ast
import os
import re
from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
import numpy as np
from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc
def load_inputs_from_input_arg_string(inputs_str, input_exprs_str, input_examples_str):
    """Parses input arg strings and create inputs feed_dict.

  Parses '--inputs' string for inputs to be loaded from file, and parses
  '--input_exprs' string for inputs to be evaluated from python expression.
  '--input_examples' string for inputs to be created from tf.example feature
  dictionary list.

  Args:
    inputs_str: A string that specified where to load inputs. Each input is
        separated by semicolon.
        * For each input key:
            '<input_key>=<filename>' or
            '<input_key>=<filename>[<variable_name>]'
        * The optional 'variable_name' key will be set to None if not specified.
        * File specified by 'filename' will be loaded using numpy.load. Inputs
            can be loaded from only .npy, .npz or pickle files.
        * The "[variable_name]" key is optional depending on the input file type
            as descripted in more details below.
        When loading from a npy file, which always contains a numpy ndarray, the
        content will be directly assigned to the specified input tensor. If a
        variable_name is specified, it will be ignored and a warning will be
        issued.
        When loading from a npz zip file, user can specify which variable within
        the zip file to load for the input tensor inside the square brackets. If
        nothing is specified, this function will check that only one file is
        included in the zip and load it for the specified input tensor.
        When loading from a pickle file, if no variable_name is specified in the
        square brackets, whatever that is inside the pickle file will be passed
        to the specified input tensor, else SavedModel CLI will assume a
        dictionary is stored in the pickle file and the value corresponding to
        the variable_name will be used.
    input_exprs_str: A string that specifies python expressions for inputs.
        * In the format of: '<input_key>=<python expression>'.
        * numpy module is available as np.
    input_examples_str: A string that specifies tf.Example with dictionary.
        * In the format of: '<input_key>=<[{feature:value list}]>'

  Returns:
    A dictionary that maps input tensor keys to numpy ndarrays.

  Raises:
    RuntimeError: An error when a key is specified, but the input file contains
        multiple numpy ndarrays, none of which matches the given key.
    RuntimeError: An error when no key is specified, but the input file contains
        more than one numpy ndarrays.
  """
    tensor_key_feed_dict = {}
    inputs = preprocess_inputs_arg_string(inputs_str)
    input_exprs = preprocess_input_exprs_arg_string(input_exprs_str)
    input_examples = preprocess_input_examples_arg_string(input_examples_str)
    for input_tensor_key, (filename, variable_name) in inputs.items():
        data = np.load(file_io.FileIO(filename, mode='rb'), allow_pickle=True)
        if variable_name:
            if isinstance(data, np.ndarray):
                logging.warn('Input file %s contains a single ndarray. Name key "%s" ignored.' % (filename, variable_name))
                tensor_key_feed_dict[input_tensor_key] = data
            elif variable_name in data:
                tensor_key_feed_dict[input_tensor_key] = data[variable_name]
            else:
                raise RuntimeError('Input file %s does not contain variable with name "%s".' % (filename, variable_name))
        elif isinstance(data, np.lib.npyio.NpzFile):
            variable_name_list = data.files
            if len(variable_name_list) != 1:
                raise RuntimeError('Input file %s contains more than one ndarrays. Please specify the name of ndarray to use.' % filename)
            tensor_key_feed_dict[input_tensor_key] = data[variable_name_list[0]]
        else:
            tensor_key_feed_dict[input_tensor_key] = data
    for input_tensor_key, py_expr_evaluated in input_exprs.items():
        if input_tensor_key in tensor_key_feed_dict:
            logging.warn('input_key %s has been specified with both --inputs and --input_exprs options. Value in --input_exprs will be used.' % input_tensor_key)
        tensor_key_feed_dict[input_tensor_key] = py_expr_evaluated
    for input_tensor_key, example in input_examples.items():
        if input_tensor_key in tensor_key_feed_dict:
            logging.warn('input_key %s has been specified in multiple options. Value in --input_examples will be used.' % input_tensor_key)
        tensor_key_feed_dict[input_tensor_key] = example
    return tensor_key_feed_dict