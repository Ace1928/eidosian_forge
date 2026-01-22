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
def get_affine(self):
    """Get the affine transform from the header information.

        MGH format doesn't store the transform directly. Instead it's gleaned
        from the zooms ( delta ), direction cosines ( Mdc ), RAS centers (
        Pxyz_c ) and the dimensions.
        """
    hdr = self._structarr
    MdcD = hdr['Mdc'].T * hdr['delta']
    vol_center = MdcD.dot(hdr['dims'][:3]) / 2
    return from_matvec(MdcD, hdr['Pxyz_c'] - vol_center)