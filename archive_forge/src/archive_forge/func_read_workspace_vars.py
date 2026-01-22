import os.path
import io
from scipy.io.matlab._mio5 import MatFile5Reader
def read_workspace_vars(fname):
    fp = open(fname, 'rb')
    rdr = MatFile5Reader(fp, struct_as_record=True)
    vars = rdr.get_variables()
    fws = vars['__function_workspace__']
    ws_bs = io.BytesIO(fws.tobytes())
    ws_bs.seek(2)
    rdr.mat_stream = ws_bs
    mi = rdr.mat_stream.read(2)
    rdr.byte_order = mi == b'IM' and '<' or '>'
    rdr.mat_stream.read(4)
    mdict = read_minimat_vars(rdr)
    fp.close()
    return mdict