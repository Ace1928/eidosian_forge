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
def writehdr_to(self, fileobj):
    """Write header to fileobj

        Write starts at the beginning.

        Parameters
        ----------
        fileobj : file-like object
           Should implement ``write`` and ``seek`` method

        Returns
        -------
        None
        """
    hdr_nofooter = np.ndarray((), dtype=self._hdrdtype, buffer=self.binaryblock)
    fileobj.seek(0)
    fileobj.write(hdr_nofooter.tobytes())