from hashlib import md5
import array
import re
def old_combined_hash(mfld):
    hash = str(' &and& '.join([old_basic_hash(mfld)] + cover_hash(mfld, (2, 3))))
    return hash.encode('utf8')