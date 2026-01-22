from urllib import parse as urlparse
def url_with_filters(url, filters=None):
    """Add a percent-encoded string of filters (a dict) to a base url."""
    if filters:
        filters = [(encode(k), encode(v)) for k, v in filters.items()]
        urlencoded_filters = urlparse.urlencode(filters)
        url = urlparse.urljoin(url, '?' + urlencoded_filters)
    return url