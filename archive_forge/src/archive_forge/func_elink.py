import time
import warnings
import io
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from Bio._utils import function_with_previous
def elink(**keywds):
    """Check for linked external articles and return a handle.

    ELink checks for the existence of an external or Related Articles link
    from a list of one or more primary IDs;  retrieves IDs and relevancy
    scores for links to Entrez databases or Related Articles; creates a
    hyperlink to the primary LinkOut provider for a specific ID and
    database, or lists LinkOut URLs and attributes for multiple IDs.

    See the online documentation for an explanation of the parameters:
    http://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ELink

    Note that ELink treats the "id" parameter differently than the other
    tools when multiple values are given. You should generally pass multiple
    UIDs as a list of strings or integers. This will provide a "one-to-one"
    mapping from source database UIDs to destination database UIDs in the
    result. If multiple source UIDs are passed as a single comma-delimited
    string all destination UIDs will be mixed together in the result.

    This example finds articles related to the Biopython application
    note's entry in the PubMed database:

    >>> from Bio import Entrez
    >>> Entrez.email = "Your.Name.Here@example.org"
    >>> pmid = "19304878"
    >>> handle = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pubmed")
    >>> record = Entrez.read(handle)
    >>> handle.close()
    >>> print(record[0]["LinkSetDb"][0]["LinkName"])
    pubmed_pubmed
    >>> linked = [link["Id"] for link in record[0]["LinkSetDb"][0]["Link"]]
    >>> "14630660" in linked
    True

    This is explained in much more detail in the Biopython Tutorial.

    :returns: Handle to the results, by default in XML format.
    :raises urllib.error.URLError: If there's a network error.
    """
    cgi = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi'
    variables = {}
    variables.update(keywds)
    request = _build_request(cgi, variables, join_ids=False)
    return _open(request)