import unittest
from nltk.corpus import brown
from nltk.jsontags import JSONTaggedDecoder, JSONTaggedEncoder
from nltk.tag import (
from nltk.tag.brill import nltkdemo18
def test_brill_tagger(self):
    trainer = BrillTaggerTrainer(self.default_tagger, nltkdemo18(), deterministic=True)
    tagger = trainer.train(self.corpus, max_rules=30)
    encoded = self.encoder.encode(tagger)
    decoded = self.decoder.decode(encoded)
    self.assertEqual(repr(tagger._initial_tagger), repr(decoded._initial_tagger))
    self.assertEqual(tagger._rules, decoded._rules)
    self.assertEqual(tagger._training_stats, decoded._training_stats)