import re
from google.protobuf import text_format # pylint: disable=relative-import
def read_prototxt(fname):
    """Return a caffe_pb2.NetParameter object that defined in a prototxt file
    """
    proto = caffe_pb2.NetParameter()
    with open(fname, 'r') as f:
        text_format.Merge(str(f.read()), proto)
    return proto