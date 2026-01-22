import gzip
import struct
from collections import deque
from os.path import splitext
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.quaternions import Quaternions
def read_lammps_dump_binary(fileobj, index=-1, colnames=None, intformat='SMALLBIG', **kwargs):
    """Read binary dump-files (after binary2txt.cpp from lammps/tools)

    :param fileobj: file-stream containing the binary lammps data
    :param index: integer or slice object (default: get the last timestep)
    :param colnames: data is columns and identified by a header
    :param intformat: lammps support different integer size.  Parameter set     at compile-time and can unfortunately not derived from data file
    :returns: list of Atoms-objects
    :rtype: list
    """
    tagformat, bigformat = dict(SMALLSMALL=('i', 'i'), SMALLBIG=('i', 'q'), BIGBIG=('q', 'q'))[intformat]
    index_end = get_max_index(index)
    if not colnames:
        colnames = ['id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz']
    images = []

    def read_variables(string):
        obj_len = struct.calcsize(string)
        data_obj = fileobj.read(obj_len)
        if obj_len != len(data_obj):
            raise EOFError
        return struct.unpack(string, data_obj)
    while True:
        try:
            magic_string = None
            ntimestep, = read_variables('=' + bigformat)
            if ntimestep < 0:
                magic_string_len = -ntimestep
                magic_string = b''.join(read_variables('=' + str(magic_string_len) + 'c'))
                endian, = read_variables('=i')
                revision, = read_variables('=i')
                ntimestep, = read_variables('=' + bigformat)
            n_atoms, triclinic = read_variables('=' + bigformat + 'i')
            boundary = read_variables('=6i')
            diagdisp = read_variables('=6d')
            if triclinic != 0:
                offdiag = read_variables('=3d')
            else:
                offdiag = (0.0,) * 3
            size_one, = read_variables('=i')
            if len(colnames) != size_one:
                raise ValueError('Provided columns do not match binary file')
            if magic_string and revision > 1:
                units_str_len, = read_variables('=i')
                if units_str_len > 0:
                    _ = b''.join(read_variables('=' + str(units_str_len) + 'c'))
                flag, = read_variables('=c')
                if flag != b'\x00':
                    time, = read_variables('=d')
                columns_str_len, = read_variables('=i')
                _ = b''.join(read_variables('=' + str(columns_str_len) + 'c'))
            nchunk, = read_variables('=i')
            pbc = np.sum(np.array(boundary).reshape((3, 2)), axis=1) == 0
            cell, celldisp = construct_cell(diagdisp, offdiag)
            data = []
            for _ in range(nchunk):
                n_data, = read_variables('=i')
                data += read_variables('=' + str(n_data) + 'd')
            data = np.array(data).reshape((-1, size_one))
            out_atoms = lammps_data_to_ase_atoms(data=data, colnames=colnames, cell=cell, celldisp=celldisp, pbc=pbc, **kwargs)
            images.append(out_atoms)
            if len(images) > index_end >= 0:
                break
        except EOFError:
            break
    return images[index]