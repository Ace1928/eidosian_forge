import io
import time
from urllib.request import urlopen
from urllib.parse import quote
from typing import Dict, List
from Bio._utils import function_with_previous
def search_count(db, query):
    """Call TogoWS search count to see how many matches a search gives.

    Arguments:
     - db - database (string), see http://togows.dbcls.jp/search
     - query - search term (string)

    You could then use the count to download a large set of search results in
    batches using the offset and limit options to Bio.TogoWS.search(). In
    general however the Bio.TogoWS.search_iter() function is simpler to use.
    """
    global _search_db_names
    if _search_db_names is None:
        _search_db_names = _get_fields(_BASE_URL + '/search')
    if db not in _search_db_names:
        import warnings
        warnings.warn("TogoWS search does not officially support database '%s'. See %s/search/ for options." % (db, _BASE_URL))
    url = _BASE_URL + f'/search/{db}/{quote(query)}/count'
    handle = _open(url)
    data = handle.read()
    handle.close()
    if not data:
        raise ValueError(f'TogoWS returned no data from URL {url}')
    try:
        return int(data.strip())
    except ValueError:
        raise ValueError(f'Expected an integer from URL {url}, got: {data!r}') from None