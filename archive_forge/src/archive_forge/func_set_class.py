from __future__ import absolute_import
import os
def set_class(self, Class):
    """Setter for 'class', since an attribute reference is an error."""
    self.Set(CLASS, Class)