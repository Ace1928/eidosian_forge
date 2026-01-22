import copy
from boto.exception import TooManyRecordsException
from boto.route53.record import ResourceRecordSets
from boto.route53.status import Status
def get_nameservers(self):
    """ Get the list of nameservers for this zone."""
    ns = self.find_records(self.name, 'NS')
    if ns is not None:
        ns = ns.resource_records
    return ns