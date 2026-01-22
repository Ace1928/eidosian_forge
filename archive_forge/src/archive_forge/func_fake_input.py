from Xlib import X
from Xlib.protocol import rq
def fake_input(self, event_type, detail=0, time=X.CurrentTime, root=X.NONE, x=0, y=0):
    FakeInput(display=self.display, opcode=self.display.get_extension_major(extname), event_type=event_type, detail=detail, time=time, root=root, x=x, y=y)