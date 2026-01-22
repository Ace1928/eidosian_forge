import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
def toggle_editable(self):
    """
        Change whether the grid is editable or not, without rebuilding
        the entire grid widget.
        """
    self.change_grid_option('editable', not self.grid_options['editable'])