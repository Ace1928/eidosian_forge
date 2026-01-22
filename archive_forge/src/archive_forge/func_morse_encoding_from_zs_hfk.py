from .links_base import Strand, Crossing, Link
import random
import collections
def morse_encoding_from_zs_hfk(link):
    """
    >>> me = morse_encoding_from_zs_hfk(Link('K8n1'))
    >>> me.width
    6
    """
    from knot_floer_homology import pd_to_morse
    pd = 'PD: ' + repr(link.PD_code())
    zs_morse = pd_to_morse(pd)
    ans = MorseEncoding(zs_morse['events'])
    assert zs_morse['girth'] == ans.width
    return ans