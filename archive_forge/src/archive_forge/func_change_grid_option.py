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
def change_grid_option(self, option_name, option_value):
    """
        Change a SlickGrid grid option without rebuilding the entire grid
        widget. Not all options are supported at this point so this
        method should be considered experimental.

        Parameters
        ----------
        option_name : str
            The name of the grid option to be changed.
        option_value : str
            The new value for the grid option.
        """
    self.grid_options[option_name] = option_value
    self.send({'type': 'change_grid_option', 'option_name': option_name, 'option_value': option_value})