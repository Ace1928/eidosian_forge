from .links_base import Strand, Crossing, Link
import random
import collections
def test_morse_machine(link):
    E = link.exterior()
    exhaust = MorseExhaustion(link, link.crossings[0])
    encoding = MorseEncoding(exhaust)
    new_link = encoding.link()
    new_E = new_link.exterior()
    return E.is_isometric_to(new_E)