from matplotlib import _api, backend_tools, cbook, widgets
def get_tool_keymap(self, name):
    """
        Return the keymap associated with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.

        Returns
        -------
        list of str
            List of keys associated with the tool.
        """
    keys = [k for k, i in self._keys.items() if i == name]
    return keys