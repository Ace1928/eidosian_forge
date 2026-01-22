from __future__ import annotations
from .analyze import AnalyzeImage
from .brikhead import AFNIImage
from .cifti2 import Cifti2Image
from .dataobj_images import DataobjImage
from .filebasedimages import FileBasedImage
from .freesurfer import MGHImage
from .gifti import GiftiImage
from .minc1 import Minc1Image
from .minc2 import Minc2Image
from .nifti1 import Nifti1Image, Nifti1Pair
from .nifti2 import Nifti2Image, Nifti2Pair
from .parrec import PARRECImage
from .spm2analyze import Spm2AnalyzeImage
from .spm99analyze import Spm99AnalyzeImage
True if spatial image axes for `img` always precede other axes

    Parameters
    ----------
    img : object
        Image object implementing at least ``shape`` attribute.

    Returns
    -------
    spatial_axes_first : bool
        True if image only has spatial axes (number of axes < 4) or image type
        known to have spatial axes preceding other axes.
    