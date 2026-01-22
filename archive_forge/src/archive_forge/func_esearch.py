import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def esearch(db, term, **keywds):
    """Run an Entrez search and return a handle to the results.

    ESearch searches and retrieves primary IDs (for use in EFetch, ELink
    and ESummary) and term translations, and optionally retains results
    for future use in the user's environment.

    See the online documentation for an explanation of the parameters:
    http://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch

    Short example:

    >>> from Bio import Entrez
    >>> Entrez.email = "Your.Name.Here@example.org"
    >>> handle = Entrez.esearch(
    ...     db="nucleotide", retmax=10, idtype="acc",
    ...     term="opuntia[ORGN] accD 2007[Publication Date]"
    ... )
    ...
    >>> record = Entrez.read(handle)
    >>> handle.close()
    >>> int(record["Count"]) >= 2
    True
    >>> "EF590893.1" in record["IdList"]
    True
    >>> "EF590892.1" in record["IdList"]
    True

    :returns: Handle to the results, which are always in XML format.
    :raises urllib.error.URLError: If there's a network error.
    """
    cgi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    variables = {'db': db, 'term': term}
    variables.update(keywds)
    request = _build_request(cgi, variables)
    return _open(request)