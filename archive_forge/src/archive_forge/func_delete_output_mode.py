from Xlib import X
from Xlib.protocol import rq, structs
def delete_output_mode(self):
    return DeleteOutputMode(display=self.display, opcode=self.display.get_extension_major(extname), output=output, mode=mode)