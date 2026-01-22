import pytest
from numpy.testing import assert_equal
from statsmodels.tools.decorators import (cache_readonly, deprecated_alias)
@pytest.mark.parametrize('warning', [FutureWarning, UserWarning])
@pytest.mark.parametrize('remove_version', [None, '0.11'])
@pytest.mark.parametrize('msg', ['test message', None])
def test_deprecated_alias(msg, remove_version, warning):
    dummy_set = dummy_factory(msg, remove_version, warning)
    with pytest.warns(warning) as w:
        dummy_set.y = 2
        assert dummy_set.x == 2
    assert warning.__class__ is w[0].category.__class__
    dummy_get = dummy_factory(msg, remove_version, warning)
    with pytest.warns(warning) as w:
        x = dummy_get.y
        assert x == 1
    assert warning.__class__ is w[0].category.__class__
    message = str(w[0].message)
    if not msg:
        if remove_version:
            assert 'will be removed' in message
        else:
            assert 'will be removed' not in message
    else:
        assert msg in message