from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def set_margins(self, x, y, xl, xr, yt, yb):
    """Set page margins.

        Arguments:
         - x         Float(0->1), Absolute X margin as % of page
         - y         Float(0->1), Absolute Y margin as % of page
         - xl        Float(0->1), Left X margin as % of page
         - xr        Float(0->1), Right X margin as % of page
         - yt        Float(0->1), Top Y margin as % of page
         - yb        Float(0->1), Bottom Y margin as % of page

        Set the page margins as proportions of the page 0->1, and also
        set the page limits x0, y0 and xlim, ylim, and page center
        xorigin, yorigin, as well as overall page width and height
        """
    xmargin_l = xl or x
    xmargin_r = xr or x
    ymargin_top = yt or y
    ymargin_btm = yb or y
    self.x0, self.y0 = (self.pagesize[0] * xmargin_l, self.pagesize[1] * ymargin_btm)
    self.xlim, self.ylim = (self.pagesize[0] * (1 - xmargin_r), self.pagesize[1] * (1 - ymargin_top))
    self.pagewidth = self.xlim - self.x0
    self.pageheight = self.ylim - self.y0
    self.xcenter, self.ycenter = (self.x0 + self.pagewidth / 2.0, self.y0 + self.pageheight / 2.0)