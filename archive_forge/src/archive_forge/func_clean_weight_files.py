import os
import param
import numpy as np
import xarray as xr
from holoviews.core.util import get_param_values
from holoviews.core.data import XArrayInterface
from holoviews.element import Image as HvImage, QuadMesh as HvQuadMesh
from holoviews.operation.datashader import regrid
from ..element import Image, QuadMesh, is_geographic
@classmethod
def clean_weight_files(cls):
    """
        Cleans existing weight files.
        """
    deleted = []
    for f in cls._files:
        try:
            os.remove(f)
            deleted.append(f)
        except FileNotFoundError:
            pass
    print('Deleted %d weight files' % len(deleted))
    cls._files = []