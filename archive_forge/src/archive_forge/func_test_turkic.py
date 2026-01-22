from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_turkic(self):
    pairs = 'I=i;I=ı;i=İ'
    all_chars = set()
    matching = set()
    for pair in pairs.split(';'):
        ch1, ch2 = pair.split('=')
        all_chars.update((ch1, ch2))
        matching.add((ch1, ch1))
        matching.add((ch1, ch2))
        matching.add((ch2, ch1))
        matching.add((ch2, ch2))
    for ch1 in all_chars:
        for ch2 in all_chars:
            m = regex.match('(?i)\\A' + ch1 + '\\Z', ch2)
            if m:
                if (ch1, ch2) not in matching:
                    self.fail('{} matching {}'.format(ascii(ch1), ascii(ch2)))
            elif (ch1, ch2) in matching:
                self.fail('{} not matching {}'.format(ascii(ch1), ascii(ch2)))