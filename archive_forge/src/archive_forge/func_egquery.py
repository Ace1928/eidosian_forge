import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def egquery(**keywds):
    """Provide Entrez database counts for a global search.

    EGQuery provides Entrez database counts in XML for a single search
    using Global Query.

    See the online documentation for an explanation of the parameters:
    http://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EGQuery

    This quick example based on a longer version from the Biopython
    Tutorial just checks there are over 60 matches for 'Biopython'
    in PubMedCentral:

    >>> from Bio import Entrez
    >>> Entrez.email = "Your.Name.Here@example.org"
    >>> handle = Entrez.egquery(term="biopython")
    >>> record = Entrez.read(handle)
    >>> handle.close()
    >>> for row in record["eGQueryResult"]:
    ...     if "pmc" in row["DbName"]:
    ...         print(int(row["Count"]) > 60)
    True

    :returns: Handle to the results, by default in XML format.
    :raises urllib.error.URLError: If there's a network error.
    """
    cgi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/egquery.fcgi'
    variables = {}
    variables.update(keywds)
    request = _build_request(cgi, variables)
    return _open(request)