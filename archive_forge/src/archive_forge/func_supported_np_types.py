from __future__ import annotations
import io
import typing as ty
from collections.abc import Sequence
from typing import Literal
import numpy as np
from .arrayproxy import ArrayLike
from .casting import sctypes_aliases
from .dataobj_images import DataobjImage
from .filebasedimages import FileBasedHeader, FileBasedImage
from .fileholders import FileMap
from .fileslice import canonical_slicers
from .orientations import apply_orientation, inv_ornt_aff
from .viewers import OrthoSlicer3D
from .volumeutils import shape_zoom_affine
def supported_np_types(obj: HasDtype) -> set[type[np.generic]]:
    """Numpy data types that instance `obj` supports

    Parameters
    ----------
    obj : object
        Object implementing `get_data_dtype` and `set_data_dtype`.  The object
        should raise ``HeaderDataError`` for setting unsupported dtypes. The
        object will likely be a header or a :class:`SpatialImage`

    Returns
    -------
    np_types : set
        set of numpy types that `obj` supports
    """
    return _supported_np_types(obj.__class__)