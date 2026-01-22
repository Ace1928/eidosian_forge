import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@property
def entity_uuid(self):
    """Entity UUID"""
    return str(base.validate_ref_and_return_uuid(self._entity_ref, self._acl_type))