import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def describe_virtual_interfaces(self, connection_id=None, virtual_interface_id=None):
    """
        Displays all virtual interfaces for an AWS account. Virtual
        interfaces deleted fewer than 15 minutes before
        DescribeVirtualInterfaces is called are also returned. If a
        connection ID is included then only virtual interfaces
        associated with this connection will be returned. If a virtual
        interface ID is included then only a single virtual interface
        will be returned.

        A virtual interface (VLAN) transmits the traffic between the
        AWS Direct Connect location and the customer.

        If a connection ID is provided, only virtual interfaces
        provisioned on the specified connection will be returned. If a
        virtual interface ID is provided, only this particular virtual
        interface will be returned.

        :type connection_id: string
        :param connection_id: ID of the connection.
        Example: dxcon-fg5678gh

        Default: None

        :type virtual_interface_id: string
        :param virtual_interface_id: ID of the virtual interface.
        Example: dxvif-123dfg56

        Default: None

        """
    params = {}
    if connection_id is not None:
        params['connectionId'] = connection_id
    if virtual_interface_id is not None:
        params['virtualInterfaceId'] = virtual_interface_id
    return self.make_request(action='DescribeVirtualInterfaces', body=json.dumps(params))