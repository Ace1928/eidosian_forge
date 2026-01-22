import base64
import binascii
import copy
import html.entities
import re
import xml.sax.saxutils
from .html import _cp1252
from .namespaces import _base, cc, dc, georss, itunes, mediarss, psc
from .sanitizer import _sanitize_html, _HTMLSanitizer
from .util import FeedParserDict
from .urls import _urljoin, make_safe_absolute_uri, resolve_relative_uris
def unknown_starttag(self, tag, attrs):
    self.depth += 1
    attrs = [self._normalize_attributes(attr) for attr in attrs]
    attrs_d = dict(attrs)
    baseuri = attrs_d.get('xml:base', attrs_d.get('base')) or self.baseuri
    if isinstance(baseuri, bytes):
        baseuri = baseuri.decode(self.encoding, 'ignore')
    if self.baseuri:
        self.baseuri = make_safe_absolute_uri(self.baseuri, baseuri) or self.baseuri
    else:
        self.baseuri = _urljoin(self.baseuri, baseuri)
    lang = attrs_d.get('xml:lang', attrs_d.get('lang'))
    if lang == '':
        lang = None
    elif lang is None:
        lang = self.lang
    if lang:
        if tag in ('feed', 'rss', 'rdf:RDF'):
            self.feeddata['language'] = lang.replace('_', '-')
    self.lang = lang
    self.basestack.append(self.baseuri)
    self.langstack.append(lang)
    for prefix, uri in attrs:
        if prefix.startswith('xmlns:'):
            self.track_namespace(prefix[6:], uri)
        elif prefix == 'xmlns':
            self.track_namespace(None, uri)
    if self.incontent and (not self.contentparams.get('type', 'xml').endswith('xml')):
        if tag in ('xhtml:div', 'div'):
            return
        self.contentparams['type'] = 'application/xhtml+xml'
    if self.incontent and self.contentparams.get('type') == 'application/xhtml+xml':
        if tag.find(':') != -1:
            prefix, tag = tag.split(':', 1)
            namespace = self.namespaces_in_use.get(prefix, '')
            if tag == 'math' and namespace == 'http://www.w3.org/1998/Math/MathML':
                attrs.append(('xmlns', namespace))
            if tag == 'svg' and namespace == 'http://www.w3.org/2000/svg':
                attrs.append(('xmlns', namespace))
        if tag == 'svg':
            self.svgOK += 1
        return self.handle_data('<%s%s>' % (tag, self.strattrs(attrs)), escape=0)
    if tag.find(':') != -1:
        prefix, suffix = tag.split(':', 1)
    else:
        prefix, suffix = ('', tag)
    prefix = self.namespacemap.get(prefix, prefix)
    if prefix:
        prefix = prefix + '_'
    if not prefix and tag not in ('title', 'link', 'description', 'name'):
        self.intextinput = 0
    if not prefix and tag not in ('title', 'link', 'description', 'url', 'href', 'width', 'height'):
        self.inimage = 0
    methodname = '_start_' + prefix + suffix
    try:
        method = getattr(self, methodname)
        return method(attrs_d)
    except AttributeError:
        unknown_tag = prefix + suffix
        if len(attrs_d) == 0:
            return self.push(unknown_tag, 1)
        else:
            context = self._get_context()
            context[unknown_tag] = attrs_d