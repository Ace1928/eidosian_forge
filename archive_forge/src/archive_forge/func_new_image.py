from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def new_image(self, **kwargs):
    self.kwargs = kwargs
    return self.result