import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
def sanitize_css(self, text):
    """Remove potentially dangerous property declarations from CSS code.
        
        In particular, properties using the CSS ``url()`` function with a scheme
        that is not considered safe are removed:
        
        >>> sanitizer = HTMLSanitizer()
        >>> sanitizer.sanitize_css(u'''
        ...   background: url(javascript:alert("foo"));
        ...   color: #000;
        ... ''')
        ['color: #000']
        
        Also, the proprietary Internet Explorer function ``expression()`` is
        always stripped:
        
        >>> sanitizer.sanitize_css(u'''
        ...   background: #fff;
        ...   color: #000;
        ...   width: e/**/xpression(alert("foo"));
        ... ''')
        ['background: #fff', 'color: #000']
        
        :param text: the CSS text; this is expected to be `unicode` and to not
                     contain any character or numeric references
        :return: a list of declarations that are considered safe
        :rtype: `list`
        :since: version 0.4.3
        """
    decls = []
    text = self._strip_css_comments(self._replace_unicode_escapes(text))
    for decl in text.split(';'):
        decl = decl.strip()
        if not decl:
            continue
        try:
            propname, value = decl.split(':', 1)
        except ValueError:
            continue
        if not self.is_safe_css(propname.strip().lower(), value.strip()):
            continue
        is_evil = False
        if self._EXPRESSION_SEARCH(value):
            is_evil = True
        for match in self._URL_FINDITER(value):
            if not self.is_safe_uri(match.group(1)):
                is_evil = True
                break
        if not is_evil:
            decls.append(decl.strip())
    return decls