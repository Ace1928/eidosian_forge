import os
from ... import osutils
from . import wrapper
def upgrade(self):
    return wrapper.quilt_upgrade(self.tree.basedir)