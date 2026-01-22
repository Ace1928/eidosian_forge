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
def set_graph_defaults(self, **attrs):
    self.add_node(Node('graph', **attrs))