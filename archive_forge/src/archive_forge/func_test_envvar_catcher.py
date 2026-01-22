import os
import pytest
def test_envvar_catcher(nameset):
    with pytest.raises(AssertionError):
        os.environ.get('Modin_FOO', 'bar')
    with pytest.raises(AssertionError):
        'modin_qux' not in os.environ
    assert 'yay_random_name' not in os.environ
    assert os.environ[nameset]