import re
def parse_siteclass_proportions(line_floats):
    """Find proportion of alignment assigned to each class.

    For models which have multiple site classes, find the proportion of the
    alignment assigned to each class.
    """
    site_classes = {}
    if line_floats:
        for n in range(len(line_floats)):
            site_classes[n] = {'proportion': line_floats[n]}
    return site_classes