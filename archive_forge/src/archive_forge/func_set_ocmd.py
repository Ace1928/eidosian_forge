import io
import math
import os
import typing
import weakref
def set_ocmd(doc: fitz.Document, xref: int=0, ocgs: typing.Union[list, None]=None, policy: OptStr=None, ve: typing.Union[list, None]=None) -> int:
    """Create or update an OCMD object in a PDF document.

    Args:
        xref: (int) 0 for creating a new object, otherwise update existing one.
        ocgs: (list) OCG xref numbers, which shall be subject to 'policy'.
        policy: one of 'AllOn', 'AllOff', 'AnyOn', 'AnyOff' (any casing).
        ve: (list) visibility expression. Use instead of 'ocgs' with 'policy'.

    Returns:
        Xref of the created or updated OCMD.
    """
    all_ocgs = set(doc.get_ocgs().keys())

    def ve_maker(ve):
        if type(ve) not in (list, tuple) or len(ve) < 2:
            raise ValueError("bad 've' format: %s" % ve)
        if ve[0].lower() not in ('and', 'or', 'not'):
            raise ValueError('bad operand: %s' % ve[0])
        if ve[0].lower() == 'not' and len(ve) != 2:
            raise ValueError("bad 've' format: %s" % ve)
        item = '[/%s' % ve[0].title()
        for x in ve[1:]:
            if type(x) is int:
                if x not in all_ocgs:
                    raise ValueError('bad OCG %i' % x)
                item += ' %i 0 R' % x
            else:
                item += ' %s' % ve_maker(x)
        item += ']'
        return item
    text = '<</Type/OCMD'
    if ocgs and type(ocgs) in (list, tuple):
        s = set(ocgs).difference(all_ocgs)
        if s != set():
            msg = 'bad OCGs: %s' % s
            raise ValueError(msg)
        text += '/OCGs[' + ' '.join(map(lambda x: '%i 0 R' % x, ocgs)) + ']'
    if policy:
        policy = str(policy).lower()
        pols = {'anyon': 'AnyOn', 'allon': 'AllOn', 'anyoff': 'AnyOff', 'alloff': 'AllOff'}
        if policy not in ('anyon', 'allon', 'anyoff', 'alloff'):
            raise ValueError('bad policy: %s' % policy)
        text += '/P/%s' % pols[policy]
    if ve:
        text += '/VE%s' % ve_maker(ve)
    text += '>>'
    if xref == 0:
        xref = doc.get_new_xref()
    elif '/Type/OCMD' not in doc.xref_object(xref, compressed=True):
        raise ValueError('bad xref or not an OCMD')
    doc.update_object(xref, text)
    return xref