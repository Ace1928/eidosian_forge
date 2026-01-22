from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def new_image_member(self, image, member_id):
    self.image = image
    self.member_id = member_id
    return self.result