import html
import re
from collections import defaultdict
def rtuple(reldict, lcon=False, rcon=False):
    """
    Pretty print the reldict as an rtuple.
    :param reldict: a relation dictionary
    :type reldict: defaultdict
    """
    items = [class_abbrev(reldict['subjclass']), reldict['subjtext'], reldict['filler'], class_abbrev(reldict['objclass']), reldict['objtext']]
    format = '[%s: %r] %r [%s: %r]'
    if lcon:
        items = [reldict['lcon']] + items
        format = '...%r)' + format
    if rcon:
        items.append(reldict['rcon'])
        format = format + '(%r...'
    printargs = tuple(items)
    return format % printargs