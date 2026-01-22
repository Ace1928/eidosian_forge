from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
def create_blank_acl_entity_from_uri(self, acl_manager, args):
    """Validates URI argument and creates blank ACL entity"""
    entity = acl_manager.create(args.URI)
    entity.validate_input_ref()
    return entity