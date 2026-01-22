import pytest
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata, iter_fields
from urllib3.packages.six import b, u
def test_request_fields(self):
    fields = [RequestField('k', b'v', filename='somefile.txt', headers={'Content-Type': 'image/jpeg'})]
    encoded, content_type = encode_multipart_formdata(fields, boundary=BOUNDARY)
    expected = b'--' + b(BOUNDARY) + b'\r\nContent-Type: image/jpeg\r\n\r\nv\r\n--' + b(BOUNDARY) + b'--\r\n'
    assert encoded == expected