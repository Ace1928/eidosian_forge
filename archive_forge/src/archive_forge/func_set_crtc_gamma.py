from Xlib import X
from Xlib.protocol import rq, structs
def set_crtc_gamma(self, crtc, size):
    return SetCrtcGamma(display=self.display, opcode=self.display.get_extension_major(extname), crtc=crtc, size=size)