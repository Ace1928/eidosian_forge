from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_datashape_list(self):
    """
        datashape_list : datashape COMMA datashape_list
                       | datashape

        Returns a list of datashape type objects, or None.
        """
    return self.parse_homogeneous_list(self.parse_datashape, lexer.COMMA, 'Expected another datashape, ' + 'type constructor parameter ' + 'lists must have uniform type')