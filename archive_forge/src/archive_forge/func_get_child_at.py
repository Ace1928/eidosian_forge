from winappdbg import win32
def get_child_at(self, x, y, bAllowTransparency=True):
    """
        Get the child window located at the given coordinates. If no such
        window exists an exception is raised.

        @see: L{get_children}

        @type  x: int
        @param x: Horizontal coordinate.

        @type  y: int
        @param y: Vertical coordinate.

        @type  bAllowTransparency: bool
        @param bAllowTransparency: If C{True} transparent areas in windows are
            ignored, returning the window behind them. If C{False} transparent
            areas are treated just like any other area.

        @rtype:  L{Window}
        @return: Child window at the requested position, or C{None} if there
            is no window at those coordinates.
        """
    try:
        if bAllowTransparency:
            hWnd = win32.RealChildWindowFromPoint(self.get_handle(), (x, y))
        else:
            hWnd = win32.ChildWindowFromPoint(self.get_handle(), (x, y))
        if hWnd:
            return self.__get_window(hWnd)
    except WindowsError:
        pass
    return None