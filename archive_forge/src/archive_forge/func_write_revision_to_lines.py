from typing import List, Optional
from .. import lazy_regex
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError
from ..revision import Revision
from .xml_serializer import (Element, SubElement, XMLSerializer,
def write_revision_to_lines(self, rev):
    """Revision object -> xml tree"""
    lines = []
    el = b'<revision committer="%s" format="%s" inventory_sha1="%s" revision_id="%s" timestamp="%.3f"' % (encode_and_escape(rev.committer), self.revision_format_num or self.format_num, rev.inventory_sha1, encode_and_escape(rev.revision_id.decode('utf-8')), rev.timestamp)
    if rev.timezone is not None:
        el += b' timezone="%s"' % str(rev.timezone).encode('ascii')
    lines.append(el + b'>\n')
    message = encode_and_escape(escape_invalid_chars(rev.message)[0])
    lines.extend((b'<message>' + message + b'</message>\n').splitlines(True))
    if rev.parent_ids:
        lines.append(b'<parents>\n')
        for parent_id in rev.parent_ids:
            _mod_revision.check_not_reserved_id(parent_id)
            lines.append(b'<revision_ref revision_id="%s" />\n' % encode_and_escape(parent_id.decode('utf-8')))
        lines.append(b'</parents>\n')
    if rev.properties:
        preamble = b'<properties>'
        for prop_name, prop_value in sorted(rev.properties.items()):
            if prop_value:
                proplines = (preamble + b'<property name="%s">%s</property>\n' % (encode_and_escape(prop_name), encode_and_escape(escape_invalid_chars(prop_value)[0]))).splitlines(True)
            else:
                proplines = [preamble + b'<property name="%s" />\n' % (encode_and_escape(prop_name),)]
            preamble = b''
            lines.extend(proplines)
        lines.append(b'</properties>\n')
    lines.append(b'</revision>\n')
    return lines