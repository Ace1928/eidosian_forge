from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
@property
def dockerfile_dir(self):
    """Returns the directory the image is to be built from."""
    return self._dockerfile_dir