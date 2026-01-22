from Xlib import X
from Xlib.protocol import rq, structs
def set_screen_config(self, size_id, rotation, config_timestamp, rate=0, timestamp=X.CurrentTime):
    """Sets the screen to the specified size, rate, rotation and reflection.

    rate can be 0 to have the server select an appropriate rate.

    """
    return SetScreenConfig(display=self.display, opcode=self.display.get_extension_major(extname), drawable=self, timestamp=timestamp, config_timestamp=config_timestamp, size_id=size_id, rotation=rotation, rate=rate)