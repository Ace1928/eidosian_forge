import unittest
from nltk.lm.preprocessing import padded_everygram_pipeline
def test_padded_everygram_pipeline(self):
    expected_train = [[('<s>',), ('<s>', 'a'), ('a',), ('a', 'b'), ('b',), ('b', 'c'), ('c',), ('c', '</s>'), ('</s>',)]]
    expected_vocab = ['<s>', 'a', 'b', 'c', '</s>']
    train_data, vocab_data = padded_everygram_pipeline(2, [['a', 'b', 'c']])
    self.assertEqual([list(sent) for sent in train_data], expected_train)
    self.assertEqual(list(vocab_data), expected_vocab)