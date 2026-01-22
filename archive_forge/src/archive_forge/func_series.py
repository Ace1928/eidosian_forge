import os
from ... import osutils
from . import wrapper
def series(self):
    return wrapper.quilt_series(self.tree, self.series_path)