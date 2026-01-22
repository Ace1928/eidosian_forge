import unittest
from nltk.corpus import brown
from nltk.jsontags import JSONTaggedDecoder, JSONTaggedEncoder
from nltk.tag import (
from nltk.tag.brill import nltkdemo18
def test_default_tagger(self):
    encoded = self.encoder.encode(self.default_tagger)
    decoded = self.decoder.decode(encoded)
    self.assertEqual(repr(self.default_tagger), repr(decoded))
    self.assertEqual(self.default_tagger._tag, decoded._tag)