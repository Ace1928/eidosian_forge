import re
from .html import _BaseHTMLProcessor
from .urls import make_safe_absolute_uri
def replace_doctype(data):
    """Strips and replaces the DOCTYPE, returns (rss_version, stripped_data)

    rss_version may be 'rss091n' or None
    stripped_data is the same XML document with a replaced DOCTYPE
    """
    start = re.search(b'<\\w', data)
    start = start and start.start() or -1
    head, data = (data[:start + 1], data[start + 1:])
    entity_results = RE_ENTITY_PATTERN.findall(head)
    head = RE_ENTITY_PATTERN.sub(b'', head)
    doctype_results = RE_DOCTYPE_PATTERN.findall(head)
    doctype = doctype_results and doctype_results[0] or b''
    if b'netscape' in doctype.lower():
        version = 'rss091n'
    else:
        version = None
    replacement = b''
    if len(doctype_results) == 1 and entity_results:
        safe_entities = [e for e in entity_results if RE_SAFE_ENTITY_PATTERN.match(e)]
        if safe_entities:
            replacement = b'<!DOCTYPE feed [\n<!ENTITY' + b'>\n<!ENTITY '.join(safe_entities) + b'>\n]>'
    data = RE_DOCTYPE_PATTERN.sub(replacement, head) + data
    safe_entities = {k.decode('utf-8'): v.decode('utf-8') for k, v in RE_SAFE_ENTITY_PATTERN.findall(replacement)}
    return (version, data, safe_entities)