import re
def format_output(self, dct, title='', s1=''):
    """Summarise results as a nicely formatted string.

        Arguments:
         - dct is a dictionary as returned by a RestrictionBatch.search()
         - title is the title of the map.
           It must be a formatted string, i.e. you must include the line break.
         - s1 is the title separating the list of enzymes that have sites from
           those without sites.
         - s1 must be a formatted string as well.

        The format of print_that is a list.
        """
    if not dct:
        dct = self.results
    ls, nc = ([], [])
    for k, v in dct.items():
        if v:
            ls.append((k, v))
        else:
            nc.append(k)
    return self.make_format(ls, title, nc, s1)