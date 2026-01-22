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
def set_shape_files(self, file_paths):
    """Add the paths of the required image files.

        If the graph needs graphic objects to
        be used as shapes or otherwise
        those need to be in the same folder as
        the graph is going to be rendered
        from. Alternatively the absolute path to
        the files can be specified when
        including the graphics in the graph.

        The files in the location pointed to by
        the path(s) specified as arguments
        to this method will be copied to
        the same temporary location where the
        graph is going to be rendered.
        """
    if isinstance(file_paths, str):
        self.shape_files.append(file_paths)
    if isinstance(file_paths, (list, tuple)):
        self.shape_files.extend(file_paths)