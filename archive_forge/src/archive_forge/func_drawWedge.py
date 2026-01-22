from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
def drawWedge(self, wedge):
    if wedge.annular:
        self.drawPath(wedge.asPolygon())
    else:
        centerx, centery, radius, startangledegrees, endangledegrees = (wedge.centerx, wedge.centery, wedge.radius, wedge.startangledegrees, wedge.endangledegrees)
        yradius, radius1, yradius1 = wedge._xtraRadii()
        if yradius is None:
            yradius = radius
        angle = endangledegrees - startangledegrees
        path = self._canvas.beginPath()
        if (radius1 == 0 or radius1 is None) and (yradius1 == 0 or yradius1 is None):
            path.moveTo(centerx, centery)
            path.arcTo(centerx - radius, centery - yradius, centerx + radius, centery + yradius, startangledegrees, angle)
        else:
            path.arc(centerx - radius, centery - yradius, centerx + radius, centery + yradius, startangledegrees, angle)
            path.arcTo(centerx - radius1, centery - yradius1, centerx + radius1, centery + yradius1, endangledegrees, -angle)
        path.close()
        self._canvas.drawPath(path, fill=self._fill, stroke=self._stroke)