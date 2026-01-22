import copy
from boto.exception import TooManyRecordsException
from boto.route53.record import ResourceRecordSets
from boto.route53.status import Status
def get_mx(self, name, all=False):
    """
        Search this Zone for MX records that match name.

        Returns a ResourceRecord.

        If there is more than one match return all as a
        ResourceRecordSets if all is True, otherwise throws
        TooManyRecordsException.
        """
    return self.find_records(name, 'MX', all=all)