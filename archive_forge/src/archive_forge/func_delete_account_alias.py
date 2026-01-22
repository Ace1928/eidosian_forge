import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_account_alias(self, alias):
    """
        Deletes an alias for the AWS account.

        For more information on account id aliases, please see
        http://goo.gl/ToB7G

        :type alias: string
        :param alias: The alias to remove from the account.
        """
    params = {'AccountAlias': alias}
    return self.get_response('DeleteAccountAlias', params)