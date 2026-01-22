import param
from . import Dataset, util
from .dimension import ViewableElement
from .element import Element
from .layout import Layout
from .options import Store
from .overlay import NdOverlay, Overlay
from .spaces import Callable, HoloMap
@classmethod
def get_overlay_label(cls, overlay, default_label=''):
    """
        Returns a label if all the elements of an overlay agree on a
        consistent label, otherwise returns the default label.
        """
    if all((el.label == overlay.get(0).label for el in overlay)):
        return overlay.get(0).label
    else:
        return default_label