import random
from contextlib import contextmanager
import pytest
from spacy.lang.en import English
from spacy.pipeline._parser_internals.nonproj import contains_cycle
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Corpus, Example
from spacy.training.augment import (
from ..util import make_tempdir
def test_make_whitespace_variant(nlp):
    text = 'They flew to New York City.\nThen they drove to Washington, D.C.'
    words = ['They', 'flew', 'to', 'New', 'York', 'City', '.', '\n', 'Then', 'they', 'drove', 'to', 'Washington', ',', 'D.C.']
    spaces = [True, True, True, True, True, False, False, False, True, True, True, True, False, True, False]
    tags = ['PRP', 'VBD', 'IN', 'NNP', 'NNP', 'NNP', '.', '_SP', 'RB', 'PRP', 'VBD', 'IN', 'NNP', ',', 'NNP']
    lemmas = ['they', 'fly', 'to', 'New', 'York', 'City', '.', '\n', 'then', 'they', 'drive', 'to', 'Washington', ',', 'D.C.']
    heads = [1, 1, 1, 4, 5, 2, 1, 10, 10, 10, 10, 10, 11, 12, 12]
    deps = ['nsubj', 'ROOT', 'prep', 'compound', 'compound', 'pobj', 'punct', 'dep', 'advmod', 'nsubj', 'ROOT', 'prep', 'pobj', 'punct', 'appos']
    ents = ['O', '', 'O', 'B-GPE', 'I-GPE', 'I-GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'B-GPE', 'O', 'B-GPE']
    doc = Doc(nlp.vocab, words=words, spaces=spaces, tags=tags, lemmas=lemmas, heads=heads, deps=deps, ents=ents)
    assert doc.text == text
    example = Example(nlp.make_doc(text), doc)
    mod_ex = make_whitespace_variant(nlp, example, ' ', 3)
    assert mod_ex.reference.ents[0].text == 'New York City'
    mod_ex = make_whitespace_variant(nlp, example, ' ', 4)
    assert mod_ex.reference.ents[0].text == 'New  York City'
    mod_ex = make_whitespace_variant(nlp, example, ' ', 5)
    assert mod_ex.reference.ents[0].text == 'New York  City'
    mod_ex = make_whitespace_variant(nlp, example, ' ', 6)
    assert mod_ex.reference.ents[0].text == 'New York City'
    for i in range(len(doc) + 1):
        mod_ex = make_whitespace_variant(nlp, example, ' ', i)
        assert mod_ex.reference[i].is_space
        assert [t.tag_ for t in mod_ex.reference] == tags[:i] + ['_SP'] + tags[i:]
        assert [t.lemma_ for t in mod_ex.reference] == lemmas[:i] + [' '] + lemmas[i:]
        assert [t.dep_ for t in mod_ex.reference] == deps[:i] + ['dep'] + deps[i:]
        assert not mod_ex.reference.has_annotation('POS')
        assert not mod_ex.reference.has_annotation('MORPH')
        assert not contains_cycle([t.head.i for t in mod_ex.reference])
        assert len(list(doc.sents)) == 2
        if i == 0:
            assert mod_ex.reference[i].head.i == 1
        else:
            assert mod_ex.reference[i].head.i == i - 1
        for j in (3, 8, 10):
            mod_ex2 = make_whitespace_variant(nlp, mod_ex, '\t\t\n', j)
            assert not contains_cycle([t.head.i for t in mod_ex2.reference])
            assert len(list(doc.sents)) == 2
            assert mod_ex2.reference[j].head.i == j - 1
        assert len(doc.ents) == len(mod_ex.reference.ents)
        assert any((t.ent_iob == 0 for t in mod_ex.reference))
        for ent in mod_ex.reference.ents:
            assert not ent[0].is_space
            assert not ent[-1].is_space
    example.reference[0].dep_ = ''
    mod_ex = make_whitespace_variant(nlp, example, ' ', 5)
    assert mod_ex.text == example.reference.text
    example.reference[0].dep_ = 'nsubj'
    example.reference.spans['spans'] = [example.reference[0:5]]
    mod_ex = make_whitespace_variant(nlp, example, ' ', 5)
    assert mod_ex.text == example.reference.text
    del example.reference.spans['spans']
    example.reference.ents = [Span(doc, 0, 2, label='ENT', kb_id='Q123')]
    mod_ex = make_whitespace_variant(nlp, example, ' ', 5)
    assert mod_ex.text == example.reference.text