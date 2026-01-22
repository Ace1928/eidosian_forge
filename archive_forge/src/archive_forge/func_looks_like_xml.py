import re
import sys
def looks_like_xml(text):
    """Check if a doctype exists or if we have some tags."""
    if xml_decl_re.match(text):
        return True
    key = hash(text)
    try:
        return _looks_like_xml_cache[key]
    except KeyError:
        m = doctype_lookup_re.match(text)
        if m is not None:
            return True
        rv = tag_re.search(text[:1000]) is not None
        _looks_like_xml_cache[key] = rv
        return rv