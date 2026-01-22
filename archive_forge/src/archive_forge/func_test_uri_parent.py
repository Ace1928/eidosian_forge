from pyxnat import uriutil
def test_uri_parent():
    uri = uriutil.uri_parent('/projects/1/subjects/2')
    assert uri == '/projects/1/subjects'