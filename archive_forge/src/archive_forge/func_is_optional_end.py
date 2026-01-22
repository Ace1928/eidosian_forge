from __future__ import absolute_import, division, unicode_literals
from . import base
def is_optional_end(self, tagname, next):
    type = next and next['type'] or None
    if tagname in ('html', 'head', 'body'):
        return type not in ('Comment', 'SpaceCharacters')
    elif tagname in ('li', 'optgroup', 'tr'):
        if type == 'StartTag':
            return next['name'] == tagname
        else:
            return type == 'EndTag' or type is None
    elif tagname in ('dt', 'dd'):
        if type == 'StartTag':
            return next['name'] in ('dt', 'dd')
        elif tagname == 'dd':
            return type == 'EndTag' or type is None
        else:
            return False
    elif tagname == 'p':
        if type in ('StartTag', 'EmptyTag'):
            return next['name'] in ('address', 'article', 'aside', 'blockquote', 'datagrid', 'dialog', 'dir', 'div', 'dl', 'fieldset', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'hr', 'menu', 'nav', 'ol', 'p', 'pre', 'section', 'table', 'ul')
        else:
            return type == 'EndTag' or type is None
    elif tagname == 'option':
        if type == 'StartTag':
            return next['name'] in ('option', 'optgroup')
        else:
            return type == 'EndTag' or type is None
    elif tagname in ('rt', 'rp'):
        if type == 'StartTag':
            return next['name'] in ('rt', 'rp')
        else:
            return type == 'EndTag' or type is None
    elif tagname == 'colgroup':
        if type in ('Comment', 'SpaceCharacters'):
            return False
        elif type == 'StartTag':
            return next['name'] != 'colgroup'
        else:
            return True
    elif tagname in ('thead', 'tbody'):
        if type == 'StartTag':
            return next['name'] in ['tbody', 'tfoot']
        elif tagname == 'tbody':
            return type == 'EndTag' or type is None
        else:
            return False
    elif tagname == 'tfoot':
        if type == 'StartTag':
            return next['name'] == 'tbody'
        else:
            return type == 'EndTag' or type is None
    elif tagname in ('td', 'th'):
        if type == 'StartTag':
            return next['name'] in ('td', 'th')
        else:
            return type == 'EndTag' or type is None
    return False