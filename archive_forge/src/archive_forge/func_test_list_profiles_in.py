import shutil
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import pytest
from IPython.core.profileapp import list_bundled_profiles, list_profiles_in
from IPython.core.profiledir import ProfileDir
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.process import getoutput
@pytest.mark.skipif(sys.implementation.name == 'pypy' and (7, 3, 13) < sys.implementation.version < (7, 3, 16), reason='Unicode issues with scandir on PyPy, see https://github.com/pypy/pypy/issues/4860')
def test_list_profiles_in():
    td = Path(tempfile.mkdtemp(dir=TMP_TEST_DIR))
    for name in ('profile_foo', 'profile_hello', 'not_a_profile'):
        Path(td / name).mkdir(parents=True)
    if dec.unicode_paths:
        Path(td / 'profile_ünicode').mkdir(parents=True)
    with open(td / 'profile_file', 'w', encoding='utf-8') as f:
        f.write('I am not a profile directory')
    profiles = list_profiles_in(td)
    found_unicode = False
    for p in list(profiles):
        if p.endswith('nicode'):
            pd = ProfileDir.find_profile_dir_by_name(td, p)
            profiles.remove(p)
            found_unicode = True
            break
    if dec.unicode_paths:
        assert found_unicode is True
    assert set(profiles) == {'foo', 'hello'}