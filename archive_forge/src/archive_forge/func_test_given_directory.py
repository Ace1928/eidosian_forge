from os import getcwd
from os.path import realpath, abspath, dirname, isfile, join as pjoin, exists
from scipy._lib._tmpdirs import tempdir, in_tempdir, in_dir
from numpy.testing import assert_, assert_equal
def test_given_directory():
    cwd = getcwd()
    with in_dir() as tmpdir:
        assert_equal(tmpdir, abspath(cwd))
        assert_equal(tmpdir, abspath(getcwd()))
    with in_dir(MY_DIR) as tmpdir:
        assert_equal(tmpdir, MY_DIR)
        assert_equal(realpath(MY_DIR), realpath(abspath(getcwd())))
    assert_(isfile(MY_PATH))