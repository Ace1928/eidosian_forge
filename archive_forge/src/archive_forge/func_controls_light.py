from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
def controls_light(return_widget=False):
    fig = gcf()
    ambient_coefficient = ipywidgets.FloatSlider(min=0, max=1, step=0.001, value=fig.ambient_coefficient, description='ambient')
    diffuse_coefficient = ipywidgets.FloatSlider(min=0, max=1, step=0.001, value=fig.diffuse_coefficient, description='diffuse')
    specular_coefficient = ipywidgets.FloatSlider(min=0, max=1, step=0.001, value=fig.specular_coefficient, description='specular')
    specular_exponent = ipywidgets.FloatSlider(min=0, max=10, step=0.001, value=fig.specular_exponent, description='specular exp')
    ipywidgets.jslink((fig, 'ambient_coefficient'), (ambient_coefficient, 'value'))
    ipywidgets.jslink((fig, 'diffuse_coefficient'), (diffuse_coefficient, 'value'))
    ipywidgets.jslink((fig, 'specular_coefficient'), (specular_coefficient, 'value'))
    ipywidgets.jslink((fig, 'specular_exponent'), (specular_exponent, 'value'))
    widgets_bottom = [ipywidgets.HBox([ambient_coefficient, diffuse_coefficient]), ipywidgets.HBox([specular_coefficient, specular_exponent])]
    current.container.children = current.container.children + widgets_bottom
    if return_widget:
        return widgets_bottom