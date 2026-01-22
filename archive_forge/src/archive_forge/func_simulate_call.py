import io
import sys
import time
import marshal
def simulate_call(self, name):
    code = self.fake_code('profile', 0, name)
    if self.cur:
        pframe = self.cur[-2]
    else:
        pframe = None
    frame = self.fake_frame(code, pframe)
    self.dispatch['call'](self, frame, 0)