import re
import html
from paste.util import PySourceColor
def long_item_list(self, lst):
    """
        Returns true if the list contains items that are long, and should
        be more nicely formatted.
        """
    how_many = 0
    for item in lst:
        if len(repr(item)) > 40:
            how_many += 1
            if how_many >= 3:
                return True
    return False