from pyxnat import uriutil
def test_uri_nextlast():
    assert uriutil.uri_nextlast('/projects/1/subjects/2') == 'subjects'