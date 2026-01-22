import warnings
from .base import Renderer
from ..exporter import Exporter
def open_figure(self, fig, props):
    self.chart = None
    self.figwidth = int(props['figwidth'] * props['dpi'])
    self.figheight = int(props['figheight'] * props['dpi'])