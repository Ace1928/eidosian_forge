import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
def list_source_files_against_dump(dump, path_regex_allowlist=None, node_name_regex_allowlist=None):
    """Generate a list of source files with information regarding ops and tensors.

  Args:
    dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
      has been loaded.
    path_regex_allowlist: A regular-expression filter for source file path.
    node_name_regex_allowlist: A regular-expression filter for node names.

  Returns:
    A list of tuples regarding the Python source files involved in constructing
    the ops and tensors contained in `dump`. Each tuple is:
      (source_file_path, is_tf_library, num_nodes, num_tensors, num_dumps,
       first_line)

      is_tf_library: (`bool`) A guess of whether the file belongs to the
        TensorFlow Python library.
      num_nodes: How many nodes were created by lines of this source file.
        These include nodes with dumps and those without.
      num_tensors: How many Tensors were created by lines of this source file.
        These include Tensors with dumps and those without.
      num_dumps: How many debug Tensor dumps were from nodes (and Tensors)
        that were created by this source file.
      first_line: The first line number (1-based) that created any nodes or
        Tensors in this source file.

    The list is sorted by ascending order of source_file_path.

  Raises:
    ValueError: If the dump object does not have a Python graph set.
  """
    py_graph = dump.python_graph
    if not py_graph:
        raise ValueError('Cannot generate source list due to a lack of set Python graph in the dump object')
    path_to_node_names = collections.defaultdict(set)
    path_to_tensor_names = collections.defaultdict(set)
    path_to_first_line = {}
    tensor_name_to_num_dumps = {}
    path_regex = re.compile(path_regex_allowlist) if path_regex_allowlist else None
    node_name_regex = re.compile(node_name_regex_allowlist) if node_name_regex_allowlist else None
    to_skip_file_paths = set()
    for op in py_graph.get_operations():
        if node_name_regex and (not node_name_regex.match(op.name)):
            continue
        for file_path, line_number, _, _ in dump.node_traceback(op.name):
            file_path = _norm_abs_path(file_path)
            if file_path in to_skip_file_paths or (path_regex and (not path_regex.match(file_path))) or (not os.path.isfile(file_path)):
                to_skip_file_paths.add(file_path)
                continue
            path_to_node_names[file_path].add(op.name)
            if file_path in path_to_first_line:
                if path_to_first_line[file_path] > line_number:
                    path_to_first_line[file_path] = line_number
            else:
                path_to_first_line[file_path] = line_number
            for output_tensor in op.outputs:
                tensor_name = output_tensor.name
                path_to_tensor_names[file_path].add(tensor_name)
            watch_keys = dump.debug_watch_keys(op.name)
            for watch_key in watch_keys:
                node_name, output_slot, debug_op = watch_key.split(':')
                tensor_name = '%s:%s' % (node_name, output_slot)
                if tensor_name not in tensor_name_to_num_dumps:
                    tensor_name_to_num_dumps[tensor_name] = len(dump.get_tensors(node_name, int(output_slot), debug_op))
    path_to_num_dumps = {}
    for path in path_to_tensor_names:
        path_to_num_dumps[path] = sum((tensor_name_to_num_dumps.get(tensor_name, 0) for tensor_name in path_to_tensor_names[path]))
    output = []
    for file_path in path_to_node_names:
        output.append((file_path, guess_is_tensorflow_py_library(file_path), len(path_to_node_names.get(file_path, {})), len(path_to_tensor_names.get(file_path, {})), path_to_num_dumps.get(file_path, 0), path_to_first_line[file_path]))
    return sorted(output, key=lambda x: x[0])