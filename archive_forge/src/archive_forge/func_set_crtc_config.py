from Xlib import X
from Xlib.protocol import rq, structs
def set_crtc_config(self, crtc, config_timestamp, mode, rotation, timestamp=X.CurrentTime):
    return SetCrtcConfig(display=self.display, opcode=self.display.get_extension_major(extname), crtc=crtc, config_timestamp=config_timestamp, mode=mode, rotation=rotation, timestamp=timestamp)