from . import win32
def erase_data(self, mode=0, on_stderr=False):
    if mode[0] not in (2,):
        return
    handle = win32.STDOUT
    if on_stderr:
        handle = win32.STDERR
    coord_screen = win32.COORD(0, 0)
    csbi = win32.GetConsoleScreenBufferInfo(handle)
    dw_con_size = csbi.dwSize.X * csbi.dwSize.Y
    win32.FillConsoleOutputCharacter(handle, ' ', dw_con_size, coord_screen)
    win32.FillConsoleOutputAttribute(handle, self.get_attrs(), dw_con_size, coord_screen)
    win32.SetConsoleCursorPosition(handle, (coord_screen.X, coord_screen.Y))