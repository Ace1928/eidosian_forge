import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
@staticmethod
def get_acl_ref_from_entity_ref_relative(entity_ref, entity_type):
    if entity_ref:
        entity_ref = entity_ref.rstrip('/')
        return '{0}/{1}/{2}'.format(entity_type, entity_ref, ACL._resource_name)