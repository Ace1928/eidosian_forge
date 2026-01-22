import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def validate_input_ref(self):
    res_title = self._acl_type.title()
    if not self.entity_ref:
        raise ValueError('{0} href is required.'.format(res_title))
    if self._parent_entity_path in self.entity_ref:
        if '/acl' in self.entity_ref:
            raise ValueError('{0} ACL URI provided. Expecting {0} URI.'.format(res_title))
        ref_type = self._acl_type
    else:
        raise ValueError('{0} URI is not specified.'.format(res_title))
    base.validate_ref_and_return_uuid(self.entity_ref, ref_type)
    return ref_type