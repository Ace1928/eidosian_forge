from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_re_groupref(self):
    self.assertEqual(regex.match('^(\\|)?([^()]+)\\1$', '|a|')[:], ('|a|', '|', 'a'))
    self.assertEqual(regex.match('^(\\|)?([^()]+)\\1?$', 'a')[:], ('a', None, 'a'))
    self.assertEqual(regex.match('^(\\|)?([^()]+)\\1$', 'a|'), None)
    self.assertEqual(regex.match('^(\\|)?([^()]+)\\1$', '|a'), None)
    self.assertEqual(regex.match('^(?:(a)|c)(\\1)$', 'aa')[:], ('aa', 'a', 'a'))
    self.assertEqual(regex.match('^(?:(a)|c)(\\1)?$', 'c')[:], ('c', None, None))
    self.assertEqual(regex.findall('(?i)(.{1,40}?),(.{1,40}?)(?:;)+(.{1,80}).{1,40}?\\3(\\ |;)+(.{1,80}?)\\1', 'TEST, BEST; LEST ; Lest 123 Test, Best'), [('TEST', ' BEST', ' LEST', ' ', '123 ')])