from contextlib import contextmanager
from ._miobase import _get_matfile_version, docfiller
from ._mio4 import MatFile4Reader, MatFile4Writer
from ._mio5 import MatFile5Reader, MatFile5Writer
@docfiller
def mat_reader_factory(file_name, appendmat=True, **kwargs):
    """
    Create reader for matlab .mat format files.

    Parameters
    ----------
    %(file_arg)s
    %(append_arg)s
    %(load_args)s
    %(struct_arg)s

    Returns
    -------
    matreader : MatFileReader object
       Initialized instance of MatFileReader class matching the mat file
       type detected in `filename`.
    file_opened : bool
       Whether the file was opened by this routine.

    """
    byte_stream, file_opened = _open_file(file_name, appendmat)
    mjv, mnv = _get_matfile_version(byte_stream)
    if mjv == 0:
        return (MatFile4Reader(byte_stream, **kwargs), file_opened)
    elif mjv == 1:
        return (MatFile5Reader(byte_stream, **kwargs), file_opened)
    elif mjv == 2:
        raise NotImplementedError('Please use HDF reader for matlab v7.3 files, e.g. h5py')
    else:
        raise TypeError('Did not recognize version %s' % mjv)