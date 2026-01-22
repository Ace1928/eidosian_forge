import numpy as np
import holoviews as hv
import holoviews.plotting.bokeh

An example of a minimal bokeh app which can be served with:

    bokeh serve --show sine

It defines a simple DynamicMap returning a Curve of a sine wave with
frequency and phase dimensions, which can be varied using sliders.
