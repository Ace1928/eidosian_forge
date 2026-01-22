import re
def print_that(self, dct, title='', s1=''):
    """Print the output of the format_output method (OBSOLETE).

        Arguments:
         - dct is a dictionary as returned by a RestrictionBatch.search()
         - title is the title of the map.
           It must be a formatted string, i.e. you must include the line break.
         - s1 is the title separating the list of enzymes that have sites from
           those without sites.
         - s1 must be a formatted string as well.

        This method prints the output of A.format_output() and it is here
        for backwards compatibility.
        """
    print(self.format_output(dct, title, s1))