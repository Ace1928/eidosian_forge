from os import getcwd
from os.path import realpath, abspath, dirname, isfile, join as pjoin, exists
from scipy._lib._tmpdirs import tempdir, in_tempdir, in_dir
from numpy.testing import assert_, assert_equal
def test_in_tempdir():
    my_cwd = getcwd()
    with in_tempdir() as tmpdir:
        with open('test.txt', 'w') as f:
            f.write('some text')
        assert_(isfile('test.txt'))
        assert_(isfile(pjoin(tmpdir, 'test.txt')))
    assert_(not exists(tmpdir))
    assert_equal(getcwd(), my_cwd)