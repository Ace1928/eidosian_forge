import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
@pytest.mark.parametrize('use_relative', [True, False])
def test_save_load_resultfile(tmpdir, use_relative):
    """Test minimally the save/load functions for result files."""
    from shutil import copytree, rmtree
    tmpdir.chdir()
    old_use_relative = config.getboolean('execution', 'use_relative_paths')
    config.set('execution', 'use_relative_paths', use_relative)
    spc = pe.Node(StrPathConfuser(in_str='2'), name='spc')
    spc.base_dir = tmpdir.mkdir('node').strpath
    result = spc.run()
    loaded_result = load_resultfile(tmpdir.join('node').join('spc').join('result_spc.pklz').strpath)
    assert result.runtime.dictcopy() == loaded_result.runtime.dictcopy()
    assert result.inputs == loaded_result.inputs
    assert result.outputs.get() == loaded_result.outputs.get()
    copytree(tmpdir.join('node').strpath, tmpdir.join('node2').strpath)
    rmtree(tmpdir.join('node').strpath)
    if use_relative:
        loaded_result2 = load_resultfile(tmpdir.join('node2').join('spc').join('result_spc.pklz').strpath)
        assert result.runtime.dictcopy() == loaded_result2.runtime.dictcopy()
        assert result.inputs == loaded_result2.inputs
        assert loaded_result2.outputs.get() != result.outputs.get()
        newpath = result.outputs.out_path.replace('/node/', '/node2/')
        assert loaded_result2.outputs.out_path == newpath
        assert loaded_result2.outputs.out_tuple[0] == newpath
        assert loaded_result2.outputs.out_dict_path['2'] == newpath
    else:
        with pytest.raises(nib.TraitError):
            load_resultfile(tmpdir.join('node2').join('spc').join('result_spc.pklz').strpath)
    config.set('execution', 'use_relative_paths', old_use_relative)