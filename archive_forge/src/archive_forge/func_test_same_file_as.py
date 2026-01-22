from io import BytesIO
from ..fileholders import FileHolder
def test_same_file_as():
    fh = FileHolder('a_fname')
    assert fh.same_file_as(fh)
    fh2 = FileHolder('a_test')
    assert not fh.same_file_as(fh2)
    sio0 = BytesIO()
    fh3 = FileHolder('a_fname', sio0)
    fh4 = FileHolder('a_fname', sio0)
    assert fh3.same_file_as(fh4)
    assert not fh3.same_file_as(fh)
    fh5 = FileHolder(fileobj=sio0)
    fh6 = FileHolder(fileobj=sio0)
    assert fh5.same_file_as(fh6)
    assert not fh5.same_file_as(fh3)
    fh4_again = FileHolder('a_fname', sio0, pos=4)
    assert fh3.same_file_as(fh4_again)