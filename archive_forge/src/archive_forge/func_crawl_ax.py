import warnings
import io
from . import utils
import matplotlib
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
def crawl_ax(self, ax):
    """Crawl the axes and process all elements within"""
    with self.renderer.draw_axes(ax=ax, props=utils.get_axes_properties(ax)):
        for line in ax.lines:
            self.draw_line(ax, line)
        for text in ax.texts:
            self.draw_text(ax, text)
        for text, ttp in zip([ax.xaxis.label, ax.yaxis.label, ax.title], ['xlabel', 'ylabel', 'title']):
            if hasattr(text, 'get_text') and text.get_text():
                self.draw_text(ax, text, force_trans=ax.transAxes, text_type=ttp)
        for artist in ax.artists:
            if isinstance(artist, matplotlib.text.Text):
                self.draw_text(ax, artist)
        for patch in ax.patches:
            self.draw_patch(ax, patch)
        for collection in ax.collections:
            self.draw_collection(ax, collection)
        for image in ax.images:
            self.draw_image(ax, image)
        legend = ax.get_legend()
        if legend is not None:
            props = utils.get_legend_properties(ax, legend)
            with self.renderer.draw_legend(legend=legend, props=props):
                if props['visible']:
                    self.crawl_legend(ax, legend)