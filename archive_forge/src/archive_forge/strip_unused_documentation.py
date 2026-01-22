import argparse
import sys
from absl import app
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import strip_unused_lib
Removes unneeded nodes from a GraphDef file.

This script is designed to help streamline models, by taking the input and
output nodes that will be used by an application and figuring out the smallest
set of operations that are required to run for those arguments. The resulting
minimal graph is then saved out.

The advantages of running this script are:
 - You may be able to shrink the file size.
 - Operations that are unsupported on your platform but still present can be
   safely removed.
The resulting graph may not be as flexible as the original though, since any
input nodes that weren't explicitly mentioned may not be accessible any more.

An example of command-line usage is:
bazel build tensorflow/python/tools:strip_unused && \
bazel-bin/tensorflow/python/tools/strip_unused \
--input_graph=some_graph_def.pb \
--output_graph=/tmp/stripped_graph.pb \
--input_node_names=input0
--output_node_names=softmax

You can also look at strip_unused_test.py for an example of how to use it.

