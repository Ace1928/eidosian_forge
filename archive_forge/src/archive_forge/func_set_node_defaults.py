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
def set_node_defaults(self, **attrs):
    """Define default node attributes.

        These attributes only apply to nodes added to the graph after
        calling this method.
        """
    self.add_node(Node('node', **attrs))