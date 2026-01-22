from doctest import ELLIPSIS
from testtools import (
from testtools.assertions import (
from testtools.content import (
from testtools.matchers import (
def assert_that_callable(self, *args, **kwargs):
    return self.assertThat(*args, **kwargs)