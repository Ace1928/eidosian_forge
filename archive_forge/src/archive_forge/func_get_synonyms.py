import boto
from boto.compat import json
from boto.cloudsearch.optionstatus import OptionStatus
from boto.cloudsearch.optionstatus import IndexFieldStatus
from boto.cloudsearch.optionstatus import ServicePoliciesStatus
from boto.cloudsearch.optionstatus import RankExpressionStatus
from boto.cloudsearch.document import DocumentServiceConnection
from boto.cloudsearch.search import SearchConnection
def get_synonyms(self):
    """
        Return a :class:`boto.cloudsearch.option.OptionStatus` object
        representing the currently defined synonym options for
        the domain.
        """
    return OptionStatus(self, None, self.layer1.describe_synonym_options, self.layer1.update_synonym_options)