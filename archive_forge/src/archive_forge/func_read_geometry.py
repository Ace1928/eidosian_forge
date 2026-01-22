import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def read_geometry(filepath, read_metadata=False, read_stamp=False):
    """Read a triangular format Freesurfer surface mesh.

    Parameters
    ----------
    filepath : str
        Path to surface file.
    read_metadata : bool, optional
        If True, read and return metadata as key-value pairs.

        Valid keys:

        * 'head' : array of int
        * 'valid' : str
        * 'filename' : str
        * 'volume' : array of int, shape (3,)
        * 'voxelsize' : array of float, shape (3,)
        * 'xras' : array of float, shape (3,)
        * 'yras' : array of float, shape (3,)
        * 'zras' : array of float, shape (3,)
        * 'cras' : array of float, shape (3,)

    read_stamp : bool, optional
        Return the comment from the file

    Returns
    -------
    coords : numpy array
        nvtx x 3 array of vertex (x, y, z) coordinates.
    faces : numpy array
        nfaces x 3 array of defining mesh triangles.
    volume_info : OrderedDict
        Returned only if `read_metadata` is True.  Key-value pairs found in the
        geometry file.
    create_stamp : str
        Returned only if `read_stamp` is True.  The comment added by the
        program that saved the file.
    """
    volume_info = OrderedDict()
    TRIANGLE_MAGIC = 16777214
    QUAD_MAGIC = 16777215
    NEW_QUAD_MAGIC = 16777213
    with open(filepath, 'rb') as fobj:
        magic = _fread3(fobj)
        if magic in (QUAD_MAGIC, NEW_QUAD_MAGIC):
            nvert = _fread3(fobj)
            nquad = _fread3(fobj)
            fmt, div = ('>i2', 100.0) if magic == QUAD_MAGIC else ('>f4', 1.0)
            coords = np.fromfile(fobj, fmt, nvert * 3).astype(np.float64) / div
            coords = coords.reshape(-1, 3)
            quads = _fread3_many(fobj, nquad * 4)
            quads = quads.reshape(nquad, 4)
            faces = np.zeros((2 * nquad, 3), dtype=int)
            nface = 0
            for quad in quads:
                if quad[0] % 2 == 0:
                    faces[nface] = (quad[0], quad[1], quad[3])
                    nface += 1
                    faces[nface] = (quad[2], quad[3], quad[1])
                    nface += 1
                else:
                    faces[nface] = (quad[0], quad[1], quad[2])
                    nface += 1
                    faces[nface] = (quad[0], quad[2], quad[3])
                    nface += 1
        elif magic == TRIANGLE_MAGIC:
            create_stamp = fobj.readline().rstrip(b'\n').decode('utf-8')
            fobj.readline()
            vnum = np.fromfile(fobj, '>i4', 1)[0]
            fnum = np.fromfile(fobj, '>i4', 1)[0]
            coords = np.fromfile(fobj, '>f4', vnum * 3).reshape(vnum, 3)
            faces = np.fromfile(fobj, '>i4', fnum * 3).reshape(fnum, 3)
            if read_metadata:
                volume_info = _read_volume_info(fobj)
        else:
            raise ValueError('File does not appear to be a Freesurfer surface')
    coords = coords.astype(np.float64)
    ret = (coords, faces)
    if read_metadata:
        if len(volume_info) == 0:
            warnings.warn('No volume information contained in the file')
        ret += (volume_info,)
    if read_stamp:
        ret += (create_stamp,)
    return ret