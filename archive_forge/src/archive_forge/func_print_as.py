import re
def print_as(self, what='list'):
    """Print the results as specified.

        Valid format are:
            'list'      -> alphabetical order
            'number'    -> number of sites in the sequence
            'map'       -> a map representation of the sequence with the sites.

        If you want more flexibility over-ride the virtual method make_format.
        """
    if what == 'map':
        self.make_format = self._make_map
    elif what == 'number':
        self.make_format = self._make_number
    else:
        self.make_format = self._make_list