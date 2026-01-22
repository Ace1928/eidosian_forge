import collections
import hashlib
import json
import warnings
import numpy as np
from tensorflow.python.util.tf_export import keras_export
def fit_on_texts(self, texts):
    """Updates internal vocabulary based on a list of texts.

        In the case where texts contains lists,
        we assume each entry of the lists to be a token.

        Required before using `texts_to_sequences` or `texts_to_matrix`.

        Args:
            texts: can be a list of strings,
                a generator of strings (for memory-efficiency),
                or a list of list of strings.
        """
    for text in texts:
        self.document_count += 1
        if self.char_level or isinstance(text, list):
            if self.lower:
                if isinstance(text, list):
                    text = [text_elem.lower() for text_elem in text]
                else:
                    text = text.lower()
            seq = text
        elif self.analyzer is None:
            seq = text_to_word_sequence(text, filters=self.filters, lower=self.lower, split=self.split)
        else:
            seq = self.analyzer(text)
        for w in seq:
            if w in self.word_counts:
                self.word_counts[w] += 1
            else:
                self.word_counts[w] = 1
        for w in set(seq):
            self.word_docs[w] += 1
    wcounts = list(self.word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    if self.oov_token is None:
        sorted_voc = []
    else:
        sorted_voc = [self.oov_token]
    sorted_voc.extend((wc[0] for wc in wcounts))
    self.word_index = dict(zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))
    self.index_word = {c: w for w, c in self.word_index.items()}
    for w, c in list(self.word_docs.items()):
        self.index_docs[self.word_index[w]] = c