import pytest
from urllib3.fields import RequestField, format_header_param_rfc2231, guess_content_type
from urllib3.packages.six import u
def test_from_tuples_html5(self):
    field = RequestField.from_tuples(u('fieldname'), (u('filenäme'), 'data'))
    cd = field.headers['Content-Disposition']
    assert cd == u('form-data; name="fieldname"; filename="filenäme"')