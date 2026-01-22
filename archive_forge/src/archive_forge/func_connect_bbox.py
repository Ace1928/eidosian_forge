from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox
from . import axes_size as Size
from .parasite_axes import HostAxes
@staticmethod
def connect_bbox(bbox1, bbox2, loc1, loc2=None):
    """
        Construct a `.Path` connecting corner *loc1* of *bbox1* to corner
        *loc2* of *bbox2*, where parameters behave as documented as for the
        `.BboxConnector` constructor.
        """
    if isinstance(bbox1, Rectangle):
        bbox1 = TransformedBbox(Bbox.unit(), bbox1.get_transform())
    if isinstance(bbox2, Rectangle):
        bbox2 = TransformedBbox(Bbox.unit(), bbox2.get_transform())
    if loc2 is None:
        loc2 = loc1
    x1, y1 = BboxConnector.get_bbox_edge_pos(bbox1, loc1)
    x2, y2 = BboxConnector.get_bbox_edge_pos(bbox2, loc2)
    return Path([[x1, y1], [x2, y2]])