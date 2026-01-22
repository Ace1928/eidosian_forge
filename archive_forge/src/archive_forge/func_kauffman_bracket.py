from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def kauffman_bracket(link):
    """
    sage: L = Link('T(2, 3)')
    sage: kauffman_bracket(L)
    q^-2 + 1 + q^2 - q^6

    sage: U4 = Link(braid_closure=[1, -1, 2, -2, 3, -3])
    sage: kauffman_bracket(U4)
    -q^-1 - 4*q - 6*q^3 - 4*q^5 - q^7

    sage: U3 = Link([])
    sage: U3.unlinked_unknot_components = 3
    sage: kauffman_bracket(U3) == (q + q**-1)**3
    True
    """
    ans = VElement()
    if isinstance(link, exhaust.MorseEncoding):
        encoded = link
    else:
        exhaustion = exhaust.MorseExhaustion(link)
        encoded = exhaust.MorseEncoding(exhaustion)
    for event in encoded:
        if event.kind == 'cup':
            ans = ans.insert_cup(event.min)
        elif event.kind == 'cap':
            ans = ans.cap_off(event.min)
        else:
            assert event.kind == 'cross'
            if event.a < event.b:
                ans = ans.add_positive_crossing(event.min)
            else:
                ans = ans.add_negative_crossing(event.min)
    assert ans.is_multiple_of_empty_pairing()
    return ans.dict[PerfectMatching([])]