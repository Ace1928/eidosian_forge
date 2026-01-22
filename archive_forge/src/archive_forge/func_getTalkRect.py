from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Drawing, Group, String
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgetbase import Widget
def getTalkRect(self, startTime, duration, trackId, text):
    """Return shapes for a specific talk"""
    g = Group()
    y_bottom = self.scaleTime(startTime + duration)
    y_top = self.scaleTime(startTime)
    y_height = y_top - y_bottom
    if trackId is None:
        x = self._colLeftEdges[1]
        width = self.width - self._colWidths[0]
    else:
        x = self._colLeftEdges[trackId]
        width = self._colWidths[trackId]
    lab = Label()
    lab.setText(text)
    lab.setOrigin(x + 0.5 * width, y_bottom + 0.5 * y_height)
    lab.boxAnchor = 'c'
    lab.width = width
    lab.height = y_height
    lab.fontSize = 6
    r = Rect(x, y_bottom, width, y_height, fillColor=colors.cyan)
    g.add(r)
    g.add(lab)
    return g