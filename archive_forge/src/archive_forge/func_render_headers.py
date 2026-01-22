from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def render_headers(self):
    """
        Renders the headers for this request field.
        """
    lines = []
    sort_keys = ['Content-Disposition', 'Content-Type', 'Content-Location']
    for sort_key in sort_keys:
        if self.headers.get(sort_key, False):
            lines.append(u'%s: %s' % (sort_key, self.headers[sort_key]))
    for header_name, header_value in self.headers.items():
        if header_name not in sort_keys:
            if header_value:
                lines.append(u'%s: %s' % (header_name, header_value))
    lines.append(u'\r\n')
    return u'\r\n'.join(lines)