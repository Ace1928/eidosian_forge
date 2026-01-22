import argparse
import re
import sys
from absl import app
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as saver_lib
def run_main():
    """Main function of freeze_graph."""
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--input_graph', type=str, default='', help="TensorFlow 'GraphDef' file to load.")
    parser.add_argument('--input_saver', type=str, default='', help='TensorFlow saver file to load.')
    parser.add_argument('--input_checkpoint', type=str, default='', help='TensorFlow variables file to load.')
    parser.add_argument('--checkpoint_version', type=int, default=2, help='Tensorflow variable file format')
    parser.add_argument('--output_graph', type=str, default='', help="Output 'GraphDef' file name.")
    parser.add_argument('--input_binary', nargs='?', const=True, type='bool', default=False, help='Whether the input files are in binary format.')
    parser.add_argument('--output_node_names', type=str, default='', help='The name of the output nodes, comma separated.')
    parser.add_argument('--restore_op_name', type=str, default='save/restore_all', help='      The name of the master restore operator. Deprecated, unused by updated       loading code.\n      ')
    parser.add_argument('--filename_tensor_name', type=str, default='save/Const:0', help='      The name of the tensor holding the save path. Deprecated, unused by       updated loading code.\n      ')
    parser.add_argument('--clear_devices', nargs='?', const=True, type='bool', default=True, help='Whether to remove device specifications.')
    parser.add_argument('--initializer_nodes', type=str, default='', help='Comma separated list of initializer nodes to run before freezing.')
    parser.add_argument('--variable_names_whitelist', type=str, default='', help='      Comma separated list of variables to convert to constants. If specified,       only those variables will be converted to constants.      ')
    parser.add_argument('--variable_names_denylist', type=str, default='', help='      Comma separated list of variables to skip converting to constants.      ')
    parser.add_argument('--input_meta_graph', type=str, default='', help="TensorFlow 'MetaGraphDef' file to load.")
    parser.add_argument('--input_saved_model_dir', type=str, default='', help="Path to the dir with TensorFlow 'SavedModel' file and variables.")
    parser.add_argument('--saved_model_tags', type=str, default='serve', help="      Group of tag(s) of the MetaGraphDef to load, in string format,      separated by ','. For tag-set contains multiple tags, all tags       must be passed in.      ")
    flags, unparsed = parser.parse_known_args()
    my_main = lambda unused_args: main(unused_args, flags)
    app.run(main=my_main, argv=[sys.argv[0]] + unparsed)