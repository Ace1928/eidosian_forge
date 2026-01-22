from winappdbg import win32
def get_classname(self):
    """
        @rtype:  str
        @return: Window class name.

        @raise WindowsError: An error occured while processing this request.
        """
    return win32.GetClassName(self.get_handle())