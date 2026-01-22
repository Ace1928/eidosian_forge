import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def epost(db, **keywds):
    """Post a file of identifiers for future use.

    Posts a file containing a list of UIs for future use in the user's
    environment to use with subsequent search strategies.

    See the online documentation for an explanation of the parameters:
    http://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EPost

    :returns: Handle to the results.
    :raises urllib.error.URLError: If there's a network error.
    """
    cgi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/epost.fcgi'
    variables = {'db': db}
    variables.update(keywds)
    request = _build_request(cgi, variables, post=True)
    return _open(request)