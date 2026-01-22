import logging
from blazarclient import command
from blazarclient import exception
def retrieve_list(self, parsed_args):
    """Retrieve a list of resources from Blazar server."""
    blazar_client = self.get_client()
    body = self.args2body(parsed_args)
    resource_manager = getattr(blazar_client, self.resource)
    data = resource_manager.list_properties(**body)
    return data