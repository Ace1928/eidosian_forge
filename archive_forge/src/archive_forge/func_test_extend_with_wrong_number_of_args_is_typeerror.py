import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
@pytest.mark.parametrize('args', [(1, 2), (1, 2, 3, 4, 5)])
def test_extend_with_wrong_number_of_args_is_typeerror(self, d, args):
    with pytest.raises(TypeError) as err:
        d.extend(*args)
    assert 'extend() takes at most 1 positional arguments' in err.value.args[0]