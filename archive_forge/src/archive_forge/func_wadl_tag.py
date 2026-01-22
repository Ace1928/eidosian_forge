from datetime import datetime
import sys
def wadl_tag(tag_name):
    """Scope a tag name with the WADL namespace."""
    return '{http://research.sun.com/wadl/2006/10}' + tag_name