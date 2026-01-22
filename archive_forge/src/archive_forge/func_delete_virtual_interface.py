import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def delete_virtual_interface(self, virtual_interface_id):
    """
        Deletes a virtual interface.

        :type virtual_interface_id: string
        :param virtual_interface_id: ID of the virtual interface.
        Example: dxvif-123dfg56

        Default: None

        """
    params = {'virtualInterfaceId': virtual_interface_id}
    return self.make_request(action='DeleteVirtualInterface', body=json.dumps(params))