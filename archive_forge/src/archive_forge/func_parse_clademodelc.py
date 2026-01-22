import re
def parse_clademodelc(branch_type_no, line_floats, site_classes):
    """Parse results specific to the clade model C."""
    if not site_classes or len(line_floats) == 0:
        return
    for n in range(len(line_floats)):
        if site_classes[n].get('branch types') is None:
            site_classes[n]['branch types'] = {}
        site_classes[n]['branch types'][branch_type_no] = line_floats[n]
    return site_classes