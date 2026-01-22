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
@classmethod
def filespec_to_file_map(klass, filespec):
    filespec = _stringify_path(filespec)
    ' Check for compressed .mgz format, then .mgh format '
    if splitext(filespec)[1].lower() == '.mgz':
        return dict(image=FileHolder(filename=filespec))
    return super().filespec_to_file_map(filespec)