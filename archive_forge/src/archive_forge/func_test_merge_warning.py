import warnings
import pytest
@pytest.mark.parametrize('case', [{'fc': 'red'}, {'fc': 1}])
def test_merge_warning(case):
    with pytest.warns(UserWarning, match='defined as \\"never\\"'):
        style.merge({'facecolor': 'never'}, case)