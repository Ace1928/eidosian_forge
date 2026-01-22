from Xlib import X
from Xlib.protocol import rq, structs
def get_output_info(self, output, config_timestamp):
    return GetOutputInfo(display=self.display, opcode=self.display.get_extension_major(extname), output=output, config_timestamp=config_timestamp)