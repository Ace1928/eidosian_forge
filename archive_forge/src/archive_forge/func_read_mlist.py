import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def read_mlist(fileobj, endianness):
    """read (nframes, 4) matrix list array from `fileobj`

    Parameters
    ----------
    fileobj : file-like
        an open file-like object implementing ``seek`` and ``read``

    Returns
    -------
    mlist : (nframes, 4) ndarray
        matrix list is an array with ``nframes`` rows and columns:

        * 0: Matrix identifier (frame number)
        * 1: matrix data start block number (subheader followed by image data)
        * 2: Last block number of matrix (image) data
        * 3: Matrix status

            * 1: hxists - rw
            * 2: exists - ro
            * 3: matrix deleted

    Notes
    -----
    A block is 512 bytes.

    ``block_no`` in the code below is 1-based.  block 1 is the main header,
    and the mlist blocks start at block number 2.

    The 512 bytes in an mlist block contain 32 rows of the int32 (nframes,
    4) mlist matrix.

    The first row of these 32 looks like a special row.  The 4 values appear
    to be (respectively):

    * not sure - maybe negative number of mlist rows (out of 31) that are
      blank and not used in this block.  Called `nfree` but unused in CTI
      code;
    * block_no - of next set of mlist entries or 2 if no more entries. We also
      allow 1 or 0 to signal no more entries;
    * <no idea>.  Called `prvblk` in CTI code, so maybe previous block no;
    * n_rows - number of mlist rows in this block (between ?0 and 31) (called
      `nused` in CTI code).
    """
    dt = np.dtype(np.int32)
    if endianness is not native_code:
        dt = dt.newbyteorder(endianness)
    mlists = []
    mlist_index = 0
    mlist_block_no = 2
    while True:
        fileobj.seek((mlist_block_no - 1) * BLOCK_SIZE)
        dat = fileobj.read(BLOCK_SIZE)
        rows = np.ndarray(shape=(32, 4), dtype=dt, buffer=dat)
        n_unused, mlist_block_no, _, n_rows = rows[0]
        if not n_unused + n_rows == 31:
            mlist = []
            return mlist
        mlists.append(rows[1:n_rows + 1])
        mlist_index += n_rows
        if mlist_block_no <= 2:
            break
    return np.row_stack(mlists)