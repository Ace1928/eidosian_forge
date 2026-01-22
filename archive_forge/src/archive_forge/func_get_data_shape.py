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
def get_data_shape(self):
    """
        Returns data shape expected based on the CIFTI-2 header

        Any dimensions omitted in the CIFTI-2 header will be given a default size of None.
        """
    from . import cifti2_axes
    if len(self.mapped_indices) == 0:
        return ()
    base_shape = [None] * (max(self.mapped_indices) + 1)
    for mim in self:
        size = len(cifti2_axes.from_index_mapping(mim))
        for idx in mim.applies_to_matrix_dimension:
            base_shape[idx] = size
    return tuple(base_shape)