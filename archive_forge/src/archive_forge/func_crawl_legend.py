import warnings
import io
from . import utils
import matplotlib
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
def crawl_legend(self, ax, legend):
    """
        Recursively look through objects in legend children
        """
    legendElements = list(utils.iter_all_children(legend._legend_box, skipContainers=True))
    legendElements.append(legend.legendPatch)
    for child in legendElements:
        child.set_zorder(1000000.0 + child.get_zorder())
        if isinstance(child, matplotlib.patches.FancyBboxPatch):
            child.set_zorder(child.get_zorder() - 1)
        try:
            if isinstance(child, matplotlib.patches.Patch):
                self.draw_patch(ax, child, force_trans=ax.transAxes)
            elif isinstance(child, matplotlib.text.Text):
                if child.get_text() != 'None':
                    self.draw_text(ax, child, force_trans=ax.transAxes)
            elif isinstance(child, matplotlib.lines.Line2D):
                self.draw_line(ax, child, force_trans=ax.transAxes)
            elif isinstance(child, matplotlib.collections.Collection):
                self.draw_collection(ax, child, force_pathtrans=ax.transAxes)
            else:
                warnings.warn('Legend element %s not impemented' % child)
        except NotImplementedError:
            warnings.warn('Legend element %s not impemented' % child)