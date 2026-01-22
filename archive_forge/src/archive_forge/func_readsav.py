import struct
import numpy as np
import tempfile
import zlib
import warnings
def readsav(file_name, idict=None, python_dict=False, uncompressed_file_name=None, verbose=False):
    """
    Read an IDL .sav file.

    Parameters
    ----------
    file_name : str
        Name of the IDL save file.
    idict : dict, optional
        Dictionary in which to insert .sav file variables.
    python_dict : bool, optional
        By default, the object return is not a Python dictionary, but a
        case-insensitive dictionary with item, attribute, and call access
        to variables. To get a standard Python dictionary, set this option
        to True.
    uncompressed_file_name : str, optional
        This option only has an effect for .sav files written with the
        /compress option. If a file name is specified, compressed .sav
        files are uncompressed to this file. Otherwise, readsav will use
        the `tempfile` module to determine a temporary filename
        automatically, and will remove the temporary file upon successfully
        reading it in.
    verbose : bool, optional
        Whether to print out information about the save file, including
        the records read, and available variables.

    Returns
    -------
    idl_dict : AttrDict or dict
        If `python_dict` is set to False (default), this function returns a
        case-insensitive dictionary with item, attribute, and call access
        to variables. If `python_dict` is set to True, this function
        returns a Python dictionary with all variable names in lowercase.
        If `idict` was specified, then variables are written to the
        dictionary specified, and the updated dictionary is returned.

    Examples
    --------
    >>> from os.path import dirname, join as pjoin
    >>> import scipy.io as sio
    >>> from scipy.io import readsav

    Get the filename for an example .sav file from the tests/data directory.

    >>> data_dir = pjoin(dirname(sio.__file__), 'tests', 'data')
    >>> sav_fname = pjoin(data_dir, 'array_float32_1d.sav')

    Load the .sav file contents.

    >>> sav_data = readsav(sav_fname)

    Get keys of the .sav file contents.

    >>> print(sav_data.keys())
    dict_keys(['array1d'])

    Access a content with a key.

    >>> print(sav_data['array1d'])
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0.]

    """
    records = []
    if python_dict or idict:
        variables = {}
    else:
        variables = AttrDict()
    f = open(file_name, 'rb')
    signature = _read_bytes(f, 2)
    if signature != b'SR':
        raise Exception('Invalid SIGNATURE: %s' % signature)
    recfmt = _read_bytes(f, 2)
    if recfmt == b'\x00\x04':
        pass
    elif recfmt == b'\x00\x06':
        if verbose:
            print('IDL Save file is compressed')
        if uncompressed_file_name:
            fout = open(uncompressed_file_name, 'w+b')
        else:
            fout = tempfile.NamedTemporaryFile(suffix='.sav')
        if verbose:
            print(' -> expanding to %s' % fout.name)
        fout.write(b'SR\x00\x04')
        while True:
            rectype = _read_long(f)
            fout.write(struct.pack('>l', int(rectype)))
            nextrec = _read_uint32(f)
            nextrec += _read_uint32(f).astype(np.int64) * 2 ** 32
            unknown = f.read(4)
            if RECTYPE_DICT[rectype] == 'END_MARKER':
                modval = np.int64(2 ** 32)
                fout.write(struct.pack('>I', int(nextrec) % modval))
                fout.write(struct.pack('>I', int((nextrec - nextrec % modval) / modval)))
                fout.write(unknown)
                break
            pos = f.tell()
            rec_string = zlib.decompress(f.read(nextrec - pos))
            nextrec = fout.tell() + len(rec_string) + 12
            fout.write(struct.pack('>I', int(nextrec % 2 ** 32)))
            fout.write(struct.pack('>I', int((nextrec - nextrec % 2 ** 32) / 2 ** 32)))
            fout.write(unknown)
            fout.write(rec_string)
        f.close()
        f = fout
        f.seek(4)
    else:
        raise Exception('Invalid RECFMT: %s' % recfmt)
    while True:
        r = _read_record(f)
        records.append(r)
        if 'end' in r:
            if r['end']:
                break
    f.close()
    heap = {}
    for r in records:
        if r['rectype'] == 'HEAP_DATA':
            heap[r['heap_index']] = r['data']
    for r in records:
        if r['rectype'] == 'VARIABLE':
            replace, new = _replace_heap(r['data'], heap)
            if replace:
                r['data'] = new
            variables[r['varname'].lower()] = r['data']
    if verbose:
        for record in records:
            if record['rectype'] == 'TIMESTAMP':
                print('-' * 50)
                print('Date: %s' % record['date'])
                print('User: %s' % record['user'])
                print('Host: %s' % record['host'])
                break
        for record in records:
            if record['rectype'] == 'VERSION':
                print('-' * 50)
                print('Format: %s' % record['format'])
                print('Architecture: %s' % record['arch'])
                print('Operating System: %s' % record['os'])
                print('IDL Version: %s' % record['release'])
                break
        for record in records:
            if record['rectype'] == 'IDENTIFICATON':
                print('-' * 50)
                print('Author: %s' % record['author'])
                print('Title: %s' % record['title'])
                print('ID Code: %s' % record['idcode'])
                break
        for record in records:
            if record['rectype'] == 'DESCRIPTION':
                print('-' * 50)
                print('Description: %s' % record['description'])
                break
        print('-' * 50)
        print('Successfully read %i records of which:' % len(records))
        rectypes = [r['rectype'] for r in records]
        for rt in set(rectypes):
            if rt != 'END_MARKER':
                print(' - %i are of type %s' % (rectypes.count(rt), rt))
        print('-' * 50)
        if 'VARIABLE' in rectypes:
            print('Available variables:')
            for var in variables:
                print(f' - {var} [{type(variables[var])}]')
            print('-' * 50)
    if idict:
        for var in variables:
            idict[var] = variables[var]
        return idict
    else:
        return variables