from io import BytesIO
from ... import tests
from .. import pack
def record_writes(data):
    writes.append(data)
    return real_write(data)