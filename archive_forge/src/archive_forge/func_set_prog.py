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
def set_prog(self, prog):
    """Sets the default program.

        Sets the default program in charge of processing
        the dot file into a graph.
        """
    self.prog = prog