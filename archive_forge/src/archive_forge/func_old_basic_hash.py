from hashlib import md5
import array
import re
def old_basic_hash(mfld, digits=6):
    return '%%%df' % digits % mfld.volume() + ' ' + repr(mfld.homology())