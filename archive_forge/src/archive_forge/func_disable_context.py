from Xlib import X
from Xlib.protocol import rq
def disable_context(self, context):
    DisableContext(display=self.display, opcode=self.display.get_extension_major(extname), context=context)