import time
from boto.compat import json
def new_statement(self, arn, ip):
    """
        Returns a new policy statement that will allow
        access to the service described by ``arn`` by the
        ip specified in ``ip``.

        :type arn: string
        :param arn: The Amazon Resource Notation identifier for the
            service you wish to provide access to.  This would be
            either the search service or the document service.

        :type ip: string
        :param ip: An IP address or CIDR block you wish to grant access
            to.
        """
    return {'Effect': 'Allow', 'Action': '*', 'Resource': arn, 'Condition': {'IpAddress': {'aws:SourceIp': [ip]}}}