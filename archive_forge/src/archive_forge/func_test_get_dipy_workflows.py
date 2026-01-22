import pytest
from packaging.version import Version
from collections import namedtuple
from ...base import traits, File, TraitedSpec, BaseInterfaceInputSpec
from ..base import (
@pytest.mark.skipif(no_dipy(), reason='DIPY is not installed')
def test_get_dipy_workflows():
    from dipy.workflows import align
    l_wkflw = get_dipy_workflows(align)
    for name, obj in l_wkflw:
        assert name.endswith('Flow')
        assert issubclass(obj, align.Workflow)