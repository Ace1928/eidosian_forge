import os.path as op
import tempfile
from pyxnat import jsonutil
def test_dump_methods():
    jtable.__repr__()
    jtable.dumps_csv()
    jtable.dumps_json()
    jtable.dump_csv(tempfile.mkstemp()[1])
    jtable.dump_json(tempfile.mkstemp()[1])
    jtable.where(psytool_audit_parentdata_audit9='0', psytool_audit_parentdata_audit2='0', psytool_audit_parentdata_audit1='2', psytool_audit_parentdata_audit10='2')
    jtable.where('5154')
    jtable.where_not('5154')
    assert True