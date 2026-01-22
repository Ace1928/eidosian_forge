from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def roundRect(self, x, y, width, height, radius):
    """Draws a rectangle with rounded corners. The corners are
        approximately quadrants of a circle, with the given radius."""
    m = 0.4472
    xhi = (x, x + width)
    xlo, xhi = (min(xhi), max(xhi))
    yhi = (y, y + height)
    ylo, yhi = (min(yhi), max(yhi))
    if isinstance(radius, (list, tuple)):
        r = [max(0, r) for r in radius]
        if len(r) < 4:
            r += (4 - len(r)) * [0]
        self.moveTo(xlo + r[2], ylo)
        self.lineTo(xhi - r[3], ylo)
        if r[3] > 0:
            t = m * r[3]
            self.curveTo(xhi - t, ylo, xhi, ylo + t, xhi, ylo + r[3])
        self.lineTo(xhi, yhi - r[1])
        if r[1] > 0:
            t = m * r[1]
            self.curveTo(xhi, yhi - t, xhi - t, yhi, xhi - r[1], yhi)
        self.lineTo(xlo + r[0], yhi)
        if r[0] > 0:
            t = m * r[0]
            self.curveTo(xlo + t, yhi, xlo, yhi - t, xlo, yhi - r[0])
        self.lineTo(xlo, ylo + r[2])
        if r[2] > 0:
            t = m * r[2]
            self.curveTo(xlo, ylo + t, xlo + t, ylo, xlo + r[2], ylo)
    else:
        t = m * radius
        self.moveTo(xlo + radius, ylo)
        self.lineTo(xhi - radius, ylo)
        self.curveTo(xhi - t, ylo, xhi, ylo + t, xhi, ylo + radius)
        self.lineTo(xhi, yhi - radius)
        self.curveTo(xhi, yhi - t, xhi - t, yhi, xhi - radius, yhi)
        self.lineTo(xlo + radius, yhi)
        self.curveTo(xlo + t, yhi, xlo, yhi - t, xlo, yhi - radius)
        self.lineTo(xlo, ylo + radius)
        self.curveTo(xlo, ylo + t, xlo + t, ylo, xlo + radius, ylo)
    self.close()