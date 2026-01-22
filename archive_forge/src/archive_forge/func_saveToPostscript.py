from sping import colors
from sping.TK import TKCanvas
from Tkinter import *
def saveToPostscript(self):
    """Save the whole canvas to a postscript file"""
    print('Saving output to file tkCanvas.eps using sping.PS')
    import sping.PS
    can = sping.PS.PSCanvas(size=self.draw.size, name='tkCanvas.eps')
    self.spingDrawingCommands(can)
    can.save('tkCanvas.eps')