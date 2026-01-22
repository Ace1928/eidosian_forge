from os.path import splitext
import numpy as np
from ..affines import from_matvec, voxel_sizes
from ..arrayproxy import ArrayProxy, reshape_dataobj
from ..batteryrunners import BatteryRunner, Report
from ..filebasedimages import SerializableImage
from ..fileholders import FileHolder
from ..filename_parser import _stringify_path
from ..openers import ImageOpener
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..volumeutils import Recoder, array_from_file, array_to_file, endian_codes
from ..wrapstruct import LabeledWrapStruct
def set_zooms(self, zooms):
    """Set zooms into header fields

        Sets the spacing of voxels in the x, y, and z dimensions.
        For four-dimensional files, a temporal zoom (repetition time, or TR, in
        ms) may be provided as a fourth sequence element.

        Parameters
        ----------
        zooms : sequence
            sequence of floats specifying spatial and (optionally) temporal
            zooms
        """
    hdr = self._structarr
    zooms = np.asarray(zooms)
    ndims = self._ndims()
    if len(zooms) > ndims:
        raise HeaderDataError('Expecting %d zoom values' % ndims)
    if np.any(zooms[:3] <= 0):
        raise HeaderDataError(f'Spatial (first three) zooms must be positive; got {tuple(zooms[:3])}')
    hdr['delta'] = zooms[:3]
    if len(zooms) == 4:
        if zooms[3] < 0:
            raise HeaderDataError(f'TR must be non-negative; got {zooms[3]}')
        hdr['tr'] = zooms[3]