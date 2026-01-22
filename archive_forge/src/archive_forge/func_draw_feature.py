from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_feature(self, feature):
    """Return list of feature elements and list of labels for them."""
    feature_elements = []
    label_elements = []
    if feature.hide:
        return (feature_elements, label_elements)
    start, end = self._current_track_start_end()
    for locstart, locend in feature.locations:
        if locend < start:
            continue
        locstart = max(locstart, start)
        if end < locstart:
            continue
        locend = min(locend, end)
        feature_sigil, label = self.get_feature_sigil(feature, locstart, locend)
        feature_elements.append(feature_sigil)
        if label is not None:
            label_elements.append(label)
    return (feature_elements, label_elements)