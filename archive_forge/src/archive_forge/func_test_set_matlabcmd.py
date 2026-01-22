import os
import pytest
import nipype.interfaces.matlab as mlab
@pytest.mark.skipif(no_matlab, reason='matlab is not available')
def test_set_matlabcmd():
    default_script_file = clean_workspace_and_get_default_script_file()
    mi = mlab.MatlabCommand()
    mi.set_default_matlab_cmd('foo')
    assert not os.path.exists(default_script_file), 'scriptfile should not exist.'
    assert mi._default_matlab_cmd == 'foo'
    mi.set_default_matlab_cmd(matlab_cmd)