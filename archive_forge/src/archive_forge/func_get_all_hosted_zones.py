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
def get_all_hosted_zones(self, start_marker=None, zone_list=None):
    """
        Returns a Python data structure with information about all
        Hosted Zones defined for the AWS account.

        :param int start_marker: start marker to pass when fetching additional
            results after a truncated list
        :param list zone_list: a HostedZones list to prepend to results
        """
    params = {}
    if start_marker:
        params = {'marker': start_marker}
    response = self.make_request('GET', '/%s/hostedzone' % self.Version, params=params)
    body = response.read()
    boto.log.debug(body)
    if response.status >= 300:
        raise exception.DNSServerError(response.status, response.reason, body)
    e = boto.jsonresponse.Element(list_marker='HostedZones', item_marker=('HostedZone',))
    h = boto.jsonresponse.XmlHandler(e, None)
    h.parse(body)
    if zone_list:
        e['ListHostedZonesResponse']['HostedZones'].extend(zone_list)
    while 'NextMarker' in e['ListHostedZonesResponse']:
        next_marker = e['ListHostedZonesResponse']['NextMarker']
        zone_list = e['ListHostedZonesResponse']['HostedZones']
        e = self.get_all_hosted_zones(next_marker, zone_list)
    return e