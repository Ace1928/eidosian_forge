from ase import Atoms
def custom_colors(self, clr=None):
    """
        Define custom colors for some atoms. Pass a dictionary of the form
        {'Fe':'red', 'Au':'yellow'} to the function.
        To reset the map to default call the method without parameters.
        """
    if clr:
        self.colors = clr
    else:
        self.colors = {}
    self._select_atom()