import re
from urllib.parse import urljoin
import bs4
from bs4.element import Comment, NavigableString, Tag
def try_urljoin(base, url, allow_fragments=True):
    """attempts urljoin, on ValueError passes through url. Shortcuts http(s):// urls"""
    if url.startswith(('https://', 'http://')):
        return url
    try:
        url = urljoin(base, url, allow_fragments=allow_fragments)
    except ValueError:
        pass
    return url