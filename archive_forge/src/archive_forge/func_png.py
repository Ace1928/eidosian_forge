from rpy2 import robjects
from rpy2.robjects.lib import ggplot2, grdevices
from IPython import get_ipython  # type: ignore
from IPython.core.display import Image  # type: ignore
def png(self, width=700, height=500):
    """ Build an Ipython "Image" (requires iPython). """
    return image_png(self, width=width, height=height)