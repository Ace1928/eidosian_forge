import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
def test_bib1(self):
    txt = '\n    Some introductory text\n    (implicit comment)\n\n        @ARTICLE{Brett2002marsbar,\n      author = {Matthew Brett and Jean-Luc Anton and Romain Valabregue and Jean-Baptise\n                Poline},\n      title = {{Region of interest analysis using an SPM toolbox}},\n      journal = {Neuroimage},\n      year = {2002},\n      volume = {16},\n      pages = {1140--1141},\n      number = {2}\n    }\n\n    @some_entry{akey, aname = "about something",\n    another={something else}}\n    '
    res = bp.bibfile.parseString(txt)
    self.assertEqual(len(res), 3)
    res2 = bp.parse_str(txt)
    self.assertEqual(res.asList(), res2.asList())
    res3 = [r.asList()[0] for r, start, end in bp.definitions.scanString(txt)]
    self.assertEqual(res.asList(), res3)