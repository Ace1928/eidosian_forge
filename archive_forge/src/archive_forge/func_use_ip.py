import boto
from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.address import Address
from boto.ec2.blockdevicemapping import BlockDeviceMapping
from boto.ec2.image import ProductCodes
from boto.ec2.networkinterface import NetworkInterface
from boto.ec2.group import Group
import base64
def use_ip(self, ip_address, dry_run=False):
    """
        Associates an Elastic IP to the instance.

        :type ip_address: Either an instance of
            :class:`boto.ec2.address.Address` or a string.
        :param ip_address: The IP address to associate
            with the instance.

        :rtype: bool
        :return: True if successful
        """
    if isinstance(ip_address, Address):
        ip_address = ip_address.public_ip
    return self.connection.associate_address(self.id, ip_address, dry_run=dry_run)