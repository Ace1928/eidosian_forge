import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def list_saml_providers(self):
    """
        Lists the SAML providers in the account.
        This operation requires `Signature Version 4`_.
        """
    return self.get_response('ListSAMLProviders', {}, list_marker='SAMLProviderList')