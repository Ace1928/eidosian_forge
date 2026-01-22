import os
import os.path
import sys
from google.protobuf import text_format
from tensorflow.python.framework import byte_swap_tensor
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import tf_export
Writes a graph proto to a file.

  The graph is written as a text proto unless `as_text` is `False`.

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.compat.v1.Session()
  tf.io.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  or

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.compat.v1.Session()
  tf.io.write_graph(sess.graph, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_or_graph_def: A `Graph` or a `GraphDef` protocol buffer.
    logdir: Directory where to write the graph. This can refer to remote
      filesystems, such as Google Cloud Storage (GCS).
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.

  Returns:
    The path of the output proto file.
  