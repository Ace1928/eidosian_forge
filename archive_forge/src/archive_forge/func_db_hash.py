from hashlib import md5
import array
import re
def db_hash(mfld):
    return md5(combined_hash(mfld)).hexdigest()