from io import BytesIO
from ... import tests
from .. import pack
def get_reader_for(self, data):
    stream = BytesIO(data)
    reader = pack.BytesRecordReader(stream)
    return reader