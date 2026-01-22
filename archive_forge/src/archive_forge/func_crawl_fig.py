import warnings
import io
from . import utils
import matplotlib
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
def crawl_fig(self, fig):
    """Crawl the figure and process all axes"""
    with self.renderer.draw_figure(fig=fig, props=utils.get_figure_properties(fig)):
        for ax in fig.axes:
            self.crawl_ax(ax)