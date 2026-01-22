from . import win32
def fore(self, fore=None, on_stderr=False):
    if fore is None:
        fore = self._default_fore
    self._fore = fore
    self.set_console(on_stderr=on_stderr)