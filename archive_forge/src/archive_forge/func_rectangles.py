from Xlib import X
from Xlib.protocol import rq, structs
def rectangles(self, region, operation, ordering, x, y, rectangles):
    Rectangles(display=self.display, opcode=self.display.get_extension_major(extname), operation=operation, region=region, ordering=ordering, window=self.id, x=x, y=y, rectangles=rectangles)