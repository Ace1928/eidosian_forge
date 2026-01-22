import numpy as np
import holoviews as hv
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider, Button
def slider_update(attrname, old, new):
    plot.update(slider.value)