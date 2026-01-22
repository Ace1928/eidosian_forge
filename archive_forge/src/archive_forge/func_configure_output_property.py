from Xlib import X
from Xlib.protocol import rq, structs
def configure_output_property(self, output, property):
    return ConfigureOutputProperty(display=self.display, opcode=self.display.get_extension_major(extname), output=output, property=property)