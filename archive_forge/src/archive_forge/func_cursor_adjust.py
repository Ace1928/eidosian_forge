from . import win32
def cursor_adjust(self, x, y, on_stderr=False):
    handle = win32.STDOUT
    if on_stderr:
        handle = win32.STDERR
    position = self.get_position(handle)
    adjusted_position = (position.Y + y, position.X + x)
    win32.SetConsoleCursorPosition(handle, adjusted_position, adjust=False)