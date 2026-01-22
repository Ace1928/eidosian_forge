import os
import sys
from troveclient.compat import common
def resize_instance(self):
    """Resize an instance flavor"""
    self._require('id', 'flavor')
    self._pretty_print(self.dbaas.instances.resize_instance, self.id, self.flavor)