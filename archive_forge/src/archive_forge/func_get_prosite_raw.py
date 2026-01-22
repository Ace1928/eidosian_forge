import io
from urllib.request import urlopen
from urllib.error import HTTPError
def get_prosite_raw(id, cgi=None):
    """Get a text handle to a raw PROSITE or PRODOC record at ExPASy.

    The cgi argument is deprecated due to changes in the ExPASy
    website.

    >>> from Bio import ExPASy
    >>> from Bio.ExPASy import Prosite
    >>> with ExPASy.get_prosite_raw('PS00001') as handle:
    ...     record = Prosite.read(handle)
    ...
    >>> print(record.accession)
    PS00001

    This function raises a ValueError if the identifier does not exist:

    >>> handle = ExPASy.get_prosite_raw("DOES_NOT_EXIST")
    Traceback (most recent call last):
        ...
    ValueError: Failed to find entry 'DOES_NOT_EXIST' on ExPASy

    """
    handle = _open(f'https://prosite.expasy.org/{id}.txt')
    if handle.url == 'https://www.expasy.org/':
        raise ValueError(f"Failed to find entry '{id}' on ExPASy") from None
    return handle