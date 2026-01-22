from io import BytesIO
from ... import tests
from .. import pack
def make_parser_expecting_bytes_record(self):
    parser = pack.ContainerPushParser()
    parser.accept_bytes(b'Bazaar pack format 1 (introduced in 0.18)\nB')
    return parser