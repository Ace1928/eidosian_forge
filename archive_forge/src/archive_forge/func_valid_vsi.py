import sys
import re
from urllib.parse import urlparse
def valid_vsi(vsi):
    """Ensures all parts of our vsi path are valid schemes."""
    return all((p in SCHEMES for p in vsi.split('+')))