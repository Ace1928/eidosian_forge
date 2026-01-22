from kivy.config import Config
from kivy.utils import strtotuple
def collide_ignore(self, touch):
    x, y = (touch.sx, touch.sy)
    for l in self.ignore_list:
        xmin, ymin, xmax, ymax = l
        if x > xmin and x < xmax and (y > ymin) and (y < ymax):
            return True