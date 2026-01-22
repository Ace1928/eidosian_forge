from Xlib import X
from Xlib.protocol import rq
from Xlib.xobject import drawable
def create_region_from_border_clip(self):
    """Create a region of the border clip of the window, i.e. the area
    that is not clipped by the parent and any sibling windows.
    """
    rid = self.display.allocate_resource_id()
    CreateRegionFromBorderClip(display=self.display, opcode=self.display.get_extension_major(extname), region=rid, window=self)
    return rid