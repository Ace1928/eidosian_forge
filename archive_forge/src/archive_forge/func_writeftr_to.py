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
def writeftr_to(self, fileobj):
    """Write footer to fileobj

        Footer data is located after the data chunk. So move there and write.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` and ``seek`` method

        Returns
        -------
        None
        """
    ftr_loc_in_hdr = len(self.binaryblock) - self._ftrdtype.itemsize
    ftr_nd = np.ndarray((), dtype=self._ftrdtype, buffer=self.binaryblock, offset=ftr_loc_in_hdr)
    fileobj.seek(self.get_footer_offset())
    fileobj.write(ftr_nd.tobytes())