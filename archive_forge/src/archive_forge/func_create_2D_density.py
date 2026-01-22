import json
import warnings
import os
from plotly import exceptions, optional_imports
from plotly.files import PLOTLY_DIR
@staticmethod
def create_2D_density(*args, **kwargs):
    FigureFactory._deprecated('create_2D_density', 'create_2d_density')
    from plotly.figure_factory import create_2d_density
    return create_2d_density(*args, **kwargs)