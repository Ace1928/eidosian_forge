from matplotlib import _api, backend_tools, cbook, widgets
def update_keymap(self, name, key):
    """
        Set the keymap to associate with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.
        key : str or list of str
            Keys to associate with the tool.
        """
    if name not in self._tools:
        raise KeyError(f'{name!r} not in Tools')
    self._remove_keys(name)
    if isinstance(key, str):
        key = [key]
    for k in key:
        if k in self._keys:
            _api.warn_external(f'Key {k} changed from {self._keys[k]} to {name}')
        self._keys[k] = name