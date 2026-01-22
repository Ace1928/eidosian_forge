from reportlab.graphics.barcode.code39 import Standard39
from reportlab.lib import colors
from reportlab.lib.units import cm
from string import ascii_uppercase, digits as string_digits
def drawOn(self, canvas, x, y):
    """Draws some blocks around the identifier's characters."""
    BaseLTOLabel.drawOn(self, canvas, x, y)
    canvas.saveState()
    canvas.setLineWidth(self.LINEWIDTH)
    canvas.setStrokeColorRGB(0, 0, 0)
    canvas.translate(x, y)
    xblocks = (self.LABELWIDTH - self.NBBLOCKS * self.BLOCKWIDTH) / 2.0
    for i in range(self.NBBLOCKS):
        font, size = self.LABELFONT
        newfont = self.LABELFONT
        if i == self.NBBLOCKS - 1:
            part = self.label[i:]
            font, size = newfont
            size /= 2.0
            newfont = (font, size)
        else:
            part = self.label[i]
        canvas.saveState()
        canvas.translate(xblocks + i * self.BLOCKWIDTH, 0)
        if self.colored and part.isdigit():
            canvas.setFillColorRGB(*getattr(colors, self.COLORSCHEME[int(part)], colors.Color(1, 1, 1)).rgb())
        else:
            canvas.setFillColorRGB(1, 1, 1)
        canvas.rect(0, 0, self.BLOCKWIDTH, self.BLOCKHEIGHT, fill=True)
        canvas.translate((self.BLOCKWIDTH + canvas.stringWidth(part, *newfont)) / 2.0, self.BLOCKHEIGHT / 2.0)
        canvas.rotate(90.0)
        canvas.setFont(*newfont)
        canvas.setFillColorRGB(0, 0, 0)
        canvas.drawCentredString(0, 0, part)
        canvas.restoreState()
    canvas.restoreState()