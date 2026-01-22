import random
from contextlib import contextmanager
import pytest
from spacy.lang.en import English
from spacy.pipeline._parser_internals.nonproj import contains_cycle
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Corpus, Example
from spacy.training.augment import (
from ..util import make_tempdir
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_make_orth_variants(nlp):
    single = [{'tags': ['NFP'], 'variants': ['…', '...']}, {'tags': [':'], 'variants': ['-', '—', '–', '--', '---', '——']}]
    words = ['\n\n', 'A', '\t', 'B', 'a', 'b', '…', '...', '-', '—', '–', '--', '---', '——']
    tags = ['_SP', 'NN', '\t', 'NN', 'NN', 'NN', 'NFP', 'NFP', ':', ':', ':', ':', ':', ':']
    spaces = [True] * len(words)
    spaces[0] = False
    spaces[2] = False
    doc = Doc(nlp.vocab, words=words, spaces=spaces, tags=tags)
    augmenter = create_orth_variants_augmenter(level=0.2, lower=0.5, orth_variants={'single': single})
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        list(reader(nlp))
    augmenter = create_orth_variants_augmenter(level=1.0, lower=1.0, orth_variants={'single': single})
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for token in example.reference:
                assert token.text == token.text.lower()
    doc = Doc(nlp.vocab, words=words, spaces=[True] * len(words))
    augmenter = create_orth_variants_augmenter(level=1.0, lower=1.0, orth_variants={'single': single})
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for ex_token, doc_token in zip(example.reference, doc):
                assert ex_token.text == doc_token.text.lower()
    doc = Doc(nlp.vocab, words=words, spaces=[True] * len(words))
    augmenter = create_orth_variants_augmenter(level=1.0, lower=0.0, orth_variants={'single': single})
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for ex_token, doc_token in zip(example.reference, doc):
                assert ex_token.text == doc_token.text