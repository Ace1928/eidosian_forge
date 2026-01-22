from hashlib import md5
import array
import re
def basic_hash(mfld, digits=6):
    if mfld.solution_type() != 'contains degenerate tetrahedra':
        volume = '%%%df' % digits % mfld.volume()
    else:
        volume = 'degenerate'
    return volume + ' ' + repr(mfld.homology())