import io
from urllib.request import urlopen
from urllib.error import HTTPError
def get_prosite_entry(id, cgi='https://prosite.expasy.org/cgi-bin/prosite/get-prosite-entry'):
    """Get a text handle to a PROSITE entry at ExPASy in HTML format.

    >>> from Bio import ExPASy
    >>> import os
    >>> with ExPASy.get_prosite_entry('PS00001') as in_handle:
    ...     html = in_handle.read()
    ...
    >>> with open("myprositerecord.html", "w") as out_handle:
    ...     length = out_handle.write(html)
    ...
    >>> os.remove("myprositerecord.html")  # tidy up

    For a non-existing key XXX, ExPASy returns an HTML-formatted page
    containing this text: 'There is currently no PROSITE entry for'
    """
    return _open(f'{cgi}?{id}')