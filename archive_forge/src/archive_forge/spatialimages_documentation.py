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
Apply an orientation change and return a new image

        If ornt is identity transform, return the original image, unchanged

        Parameters
        ----------
        ornt : (n,2) orientation array
           orientation transform. ``ornt[N,1]` is flip of axis N of the
           array implied by `shape`, where 1 means no flip and -1 means
           flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
           there's an array ``arr`` of shape `shape`, the flip would
           correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
           the transpose that needs to be done to the implied array, as in
           ``arr.transpose(ornt[:,0])``

        Notes
        -----
        Subclasses should override this if they have additional requirements
        when re-orienting an image.
        