import hmac
import hashlib
import datetime
import requests
@classmethod
def get_canonical_querystring(cls, r):
    """
        Create the canonical query string. According to AWS, by the
        end of this function our query string values must
        be URL-encoded (space=%20) and the parameters must be sorted
        by name.

        This method assumes that the query params in `r` are *already*
        url encoded.  If they are not url encoded by the time they make
        it to this function, AWS may complain that the signature for your
        request is incorrect.

        It appears elasticsearc-py url encodes query paramaters on its own:
            https://github.com/elastic/elasticsearch-py/blob/5dfd6985e5d32ea353d2b37d01c2521b2089ac2b/elasticsearch/connection/http_requests.py#L64

        If you are using a different client than elasticsearch-py, it
        will be your responsibility to urleconde your query params before
        this method is called.
        """
    canonical_querystring = ''
    parsedurl = urlparse(r.url)
    querystring_sorted = '&'.join(sorted(parsedurl.query.split('&')))
    for query_param in querystring_sorted.split('&'):
        key_val_split = query_param.split('=', 1)
        key = key_val_split[0]
        if len(key_val_split) > 1:
            val = key_val_split[1]
        else:
            val = ''
        if key:
            if canonical_querystring:
                canonical_querystring += '&'
            canonical_querystring += u'='.join([key, val])
    return canonical_querystring