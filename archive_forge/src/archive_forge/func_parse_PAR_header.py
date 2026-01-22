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
def parse_PAR_header(fobj):
    """Parse a PAR header and aggregate all information into useful containers.

    Parameters
    ----------
    fobj : file-object
        The PAR header file object.

    Returns
    -------
    general_info : dict
        Contains all "General Information" from the header file
    image_info : ndarray
        Structured array with fields giving all "Image information" in the
        header
    """
    version, gen_dict, image_lines = _split_header(fobj)
    if version not in supported_versions:
        warnings.warn(one_line(f" PAR/REC version '{version}' is currently not supported -- making an\n            attempt to read nevertheless. Please email the NiBabel mailing\n            list, if you are interested in adding support for this version.\n            "))
    general_info = _process_gen_dict(gen_dict)
    image_defs = _process_image_lines(image_lines, version)
    return (general_info, image_defs)