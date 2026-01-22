import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
Create PARREC image from filename `filename`

        Parameters
        ----------
        filename : str
            Filename of "PAR" or "REC" file
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        permit_truncated : {False, True}, optional, keyword-only
            If False, raise an error for an image where the header shows signs
            that fewer slices / volumes were recorded than were expected.
        scaling : {'dv', 'fp'}, optional, keyword-only
            Scaling method to apply to data (see
            :meth:`PARRECHeader.get_data_scaling`).
        strict_sort : bool, optional, keyword-only
            If True, a larger number of header fields are used while sorting
            the REC data array.  This may produce a different sort order than
            `strict_sort=False`, where volumes are sorted by the order in which
            the slices appear in the .PAR file.
        