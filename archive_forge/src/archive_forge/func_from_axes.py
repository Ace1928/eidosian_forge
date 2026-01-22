import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn
import numpy as np
from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes
@classmethod
def from_axes(cls, axes):
    """
        Creates a new Cifti2 header based on the Cifti2 axes

        Parameters
        ----------
        axes : tuple of :class`.cifti2_axes.Axis`
            sequence of Cifti2 axes describing each row/column of the matrix to be stored

        Returns
        -------
        header : Cifti2Header
            new header describing the rows/columns in a format consistent with Cifti2
        """
    from . import cifti2_axes
    return cifti2_axes.to_header(axes)