import pytest
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata, iter_fields
from urllib3.packages.six import b, u
@pytest.mark.parametrize('fields', [dict(k='v', k2='v2'), [('k', 'v'), ('k2', 'v2')]])
def test_input_datastructures(self, fields):
    encoded, _ = encode_multipart_formdata(fields, boundary=BOUNDARY)
    assert encoded.count(b(BOUNDARY)) == 3