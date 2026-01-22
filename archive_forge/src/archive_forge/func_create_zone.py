from boto.route53 import exception
import random
import uuid
import xml.sax
import boto
from boto.connection import AWSAuthConnection
from boto import handler
import boto.jsonresponse
from boto.route53.record import ResourceRecordSets
from boto.route53.zone import Zone
from boto.compat import six, urllib
def create_zone(self, name, private_zone=False, vpc_id=None, vpc_region=None):
    """
        Create a new Hosted Zone.  Returns a Zone object for the newly
        created Hosted Zone.

        :type name: str
        :param name: The name of the domain. This should be a
            fully-specified domain, and should end with a final period
            as the last label indication.  If you omit the final period,
            Amazon Route 53 assumes the domain is relative to the root.
            This is the name you have registered with your DNS registrar.
            It is also the name you will delegate from your registrar to
            the Amazon Route 53 delegation servers returned in
            response to this request.

        :type private_zone: bool
        :param private_zone: Set True if creating a private hosted zone.

        :type vpc_id: str
        :param vpc_id: When creating a private hosted zone, the VPC Id to
            associate to is required.

        :type vpc_region: str
        :param vpc_region: When creating a private hosted zone, the region
            of the associated VPC is required.
        """
    zone = self.create_hosted_zone(name, private_zone=private_zone, vpc_id=vpc_id, vpc_region=vpc_region)
    return Zone(self, zone['CreateHostedZoneResponse']['HostedZone'])