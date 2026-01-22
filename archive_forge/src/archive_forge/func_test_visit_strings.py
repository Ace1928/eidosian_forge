import os
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
import pyarrow.tests.util as test_util
@pytest.mark.cython
def test_visit_strings(tmpdir):
    with tmpdir.as_cwd():
        pyx_file = 'bound_function_visit_strings.pyx'
        shutil.copyfile(os.path.join(here, pyx_file), os.path.join(str(tmpdir), pyx_file))
        setup_code = setup_template.format(pyx_file=pyx_file, compiler_opts=compiler_opts, test_ld_path=test_ld_path)
        with open('setup.py', 'w') as f:
            f.write(setup_code)
        subprocess_env = test_util.get_modified_env_with_pythonpath()
        subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'], env=subprocess_env)
    sys.path.insert(0, str(tmpdir))
    mod = __import__('bound_function_visit_strings')
    strings = ['a', 'b', 'c']
    visited = []
    mod._visit_strings(strings, visited.append)
    assert visited == strings
    with pytest.raises(ValueError, match='wtf'):

        def raise_on_b(s):
            if s == 'b':
                raise ValueError('wtf')
        mod._visit_strings(strings, raise_on_b)