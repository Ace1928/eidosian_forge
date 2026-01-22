from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_case_folding(self):
    self.assertEqual(regex.search('(?fi)ss', 'SS').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)SS', 'ss').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)SS', 'ß').span(), (0, 1))
    self.assertEqual(regex.search('(?fi)\\N{LATIN SMALL LETTER SHARP S}', 'SS').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)\\N{LATIN SMALL LIGATURE ST}', 'ST').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)ST', 'ﬆ').span(), (0, 1))
    self.assertEqual(regex.search('(?fi)ST', 'ﬅ').span(), (0, 1))
    self.assertEqual(regex.search('(?fi)SST', 'ßt').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)SST', 'sﬅ').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)SST', 'sﬆ').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)\\N{LATIN SMALL LIGATURE ST}', 'SST').span(), (1, 3))
    self.assertEqual(regex.search('(?fi)SST', 'sﬆ').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)FFI', 'ﬃ').span(), (0, 1))
    self.assertEqual(regex.search('(?fi)FFI', 'ﬀi').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)FFI', 'fﬁ').span(), (0, 2))
    self.assertEqual(regex.search('(?fi)\\N{LATIN SMALL LIGATURE FFI}', 'FFI').span(), (0, 3))
    self.assertEqual(regex.search('(?fi)\\N{LATIN SMALL LIGATURE FF}i', 'FFI').span(), (0, 3))
    self.assertEqual(regex.search('(?fi)f\\N{LATIN SMALL LIGATURE FI}', 'FFI').span(), (0, 3))
    sigma = 'Σσς'
    for ch1 in sigma:
        for ch2 in sigma:
            if not regex.match('(?fi)' + ch1, ch2):
                self.fail()
    self.assertEqual(bool(regex.search('(?iV1)ff', 'ﬀﬁ')), True)
    self.assertEqual(bool(regex.search('(?iV1)ff', 'ﬁﬀ')), True)
    self.assertEqual(bool(regex.search('(?iV1)fi', 'ﬀﬁ')), True)
    self.assertEqual(bool(regex.search('(?iV1)fi', 'ﬁﬀ')), True)
    self.assertEqual(bool(regex.search('(?iV1)fffi', 'ﬀﬁ')), True)
    self.assertEqual(bool(regex.search('(?iV1)f\\uFB03', 'ﬀﬁ')), True)
    self.assertEqual(bool(regex.search('(?iV1)ff', 'ﬀﬁ')), True)
    self.assertEqual(bool(regex.search('(?iV1)fi', 'ﬀﬁ')), True)
    self.assertEqual(bool(regex.search('(?iV1)fffi', 'ﬀﬁ')), True)
    self.assertEqual(bool(regex.search('(?iV1)f\\uFB03', 'ﬀﬁ')), True)
    self.assertEqual(bool(regex.search('(?iV1)f\\uFB01', 'ﬀi')), True)
    self.assertEqual(bool(regex.search('(?iV1)f\\uFB01', 'ﬀi')), True)
    self.assertEqual(regex.findall('(?iV0)\\m(?:word){e<=3}\\M(?<!\\m(?:word){e<=1}\\M)', 'word word2 word word3 word word234 word23 word'), ['word234', 'word23'])
    self.assertEqual(regex.findall('(?iV1)\\m(?:word){e<=3}\\M(?<!\\m(?:word){e<=1}\\M)', 'word word2 word word3 word word234 word23 word'), ['word234', 'word23'])
    self.assertEqual(regex.search('(?fi)a\\N{LATIN SMALL LIGATURE FFI}ne', '  affine  ').span(), (2, 8))
    self.assertEqual(regex.search('(?fi)a(?:\\N{LATIN SMALL LIGATURE FFI}|x)ne', '  affine  ').span(), (2, 8))
    self.assertEqual(regex.search('(?fi)a(?:\\N{LATIN SMALL LIGATURE FFI}|xy)ne', '  affine  ').span(), (2, 8))
    self.assertEqual(regex.search('(?fi)a\\L<options>ne', 'affine', options=['ﬃ']).span(), (0, 6))
    self.assertEqual(regex.search('(?fi)a\\L<options>ne', 'aﬃne', options=['ffi']).span(), (0, 4))