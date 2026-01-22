import pytest
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata, iter_fields
from urllib3.packages.six import b, u
def test_textplain(self):
    fields = [('k', ('somefile.txt', b'v'))]
    encoded, content_type = encode_multipart_formdata(fields, boundary=BOUNDARY)
    expected = b'--' + b(BOUNDARY) + b'\r\nContent-Disposition: form-data; name="k"; filename="somefile.txt"\r\nContent-Type: text/plain\r\n\r\nv\r\n--' + b(BOUNDARY) + b'--\r\n'
    assert encoded == expected
    assert content_type == 'multipart/form-data; boundary=' + str(BOUNDARY)