from winappdbg import win32
def get_style(self):
    """
        @rtype:  int
        @return: Window style mask.

        @raise WindowsError: An error occured while processing this request.
        """
    return win32.GetWindowLongPtr(self.get_handle(), win32.GWL_STYLE)