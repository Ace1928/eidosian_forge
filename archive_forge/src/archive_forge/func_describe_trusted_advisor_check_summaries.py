import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.support import exceptions
def describe_trusted_advisor_check_summaries(self, check_ids):
    """
        Returns the summaries of the results of the Trusted Advisor
        checks that have the specified check IDs. Check IDs can be
        obtained by calling DescribeTrustedAdvisorChecks.

        The response contains an array of TrustedAdvisorCheckSummary
        objects.

        :type check_ids: list
        :param check_ids: The IDs of the Trusted Advisor checks.

        """
    params = {'checkIds': check_ids}
    return self.make_request(action='DescribeTrustedAdvisorCheckSummaries', body=json.dumps(params))