import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def set_suppress_disconnected(self, val):
    """Suppress disconnected nodes in the output graph.

        This option will skip nodes in
        the graph with no incoming or outgoing
        edges. This option works also
        for subgraphs and has effect only in the
        current graph/subgraph.
        """
    self.obj_dict['suppress_disconnected'] = val