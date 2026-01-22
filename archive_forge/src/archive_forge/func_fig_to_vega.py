import warnings
import json
import random
from .base import Renderer
from ..exporter import Exporter
def fig_to_vega(fig, notebook=False):
    """Convert a matplotlib figure to vega dictionary

    if notebook=True, then return an object which will display in a notebook
    otherwise, return an HTML string.
    """
    renderer = VegaRenderer()
    Exporter(renderer).run(fig)
    vega_html = VegaHTML(renderer)
    if notebook:
        return vega_html
    else:
        return vega_html.html()