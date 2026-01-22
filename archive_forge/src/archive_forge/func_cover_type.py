from hashlib import md5
import array
import re
def cover_type(mfld):
    return re.findall('~reg~|~irr~|~cyc~', mfld.name())[-1][1:-1]