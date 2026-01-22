from reportlab.graphics.shapes import Drawing, Line, String, Group, Polygon
from reportlab.lib import colors
from ._AbstractDrawer import AbstractDrawer, draw_box, draw_arrow
from ._AbstractDrawer import draw_cut_corner_box, _stroke_and_fill_colors
from ._AbstractDrawer import intermediate_points, angle2trig, deduplicate
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import ceil
def init_fragments(self):
    """Initialize useful values for positioning diagram elements."""
    self.fragment_height = self.pageheight / self.fragments
    self.fragment_bases = ceil(self.length / self.fragments)
    self.fragment_lines = {}
    fragment_crop = (1 - self.fragment_size) / 2
    fragy = self.ylim
    for fragment in range(self.fragments):
        fragtop = fragy - fragment_crop * self.fragment_height
        fragbtm = fragy - (1 - fragment_crop) * self.fragment_height
        self.fragment_lines[fragment] = (fragbtm, fragtop)
        fragy -= self.fragment_height
    self.fragment_limits = {}
    fragment_step = self.fragment_bases
    fragment_count = 0
    for marker in range(int(self.start), int(self.end), int(fragment_step)):
        self.fragment_limits[fragment_count] = (marker, marker + fragment_step)
        fragment_count += 1