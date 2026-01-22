import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def define_analysis_scheme(self, domain_name, analysis_scheme):
    """
        Configures an analysis scheme that can be applied to a `text`
        or `text-array` field to define language-specific text
        processing options. For more information, see `Configuring
        Analysis Schemes`_ in the Amazon CloudSearch Developer Guide .

        :type domain_name: string
        :param domain_name: A string that represents the name of a domain.
            Domain names are unique across the domains owned by an account
            within an AWS region. Domain names start with a letter or number
            and can contain the following characters: a-z (lowercase), 0-9, and
            - (hyphen).

        :type analysis_scheme: dict
        :param analysis_scheme: Configuration information for an analysis
            scheme. Each analysis scheme has a unique name and specifies the
            language of the text to be processed. The following options can be
            configured for an analysis scheme: `Synonyms`, `Stopwords`,
            `StemmingDictionary`, and `AlgorithmicStemming`.

        """
    params = {'DomainName': domain_name}
    self.build_complex_param(params, 'AnalysisScheme', analysis_scheme)
    return self._make_request(action='DefineAnalysisScheme', verb='POST', path='/', params=params)