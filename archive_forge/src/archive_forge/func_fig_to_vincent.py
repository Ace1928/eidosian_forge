import warnings
from .base import Renderer
from ..exporter import Exporter
def fig_to_vincent(fig):
    """Convert a matplotlib figure to a vincent object"""
    renderer = VincentRenderer()
    exporter = Exporter(renderer)
    exporter.run(fig)
    return renderer.chart