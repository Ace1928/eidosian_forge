import json
import netaddr
import re
def fully(self):
    """Returns True if it's fully masked."""
    return self._mask == self.max_mask()