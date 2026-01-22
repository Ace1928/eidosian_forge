import itertools
import logging
import warnings
from unittest import mock
import pytest
from thinc.api import CupyOps, NumpyOps, get_current_ops
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import find_matching_language, ignore_error, raise_error, registry
from spacy.vocab import Vocab
from .util import add_vecs_to_vocab, assert_docs_equal
def test_language_from_config_before_after_init():
    name = 'test_language_from_config_before_after_init'
    ran_before = False
    ran_after = False
    ran_after_pipeline = False
    ran_before_init = False
    ran_after_init = False

    @registry.callbacks(f'{name}_before')
    def make_before_creation():

        def before_creation(lang_cls):
            nonlocal ran_before
            ran_before = True
            assert lang_cls is English
            lang_cls.Defaults.foo = 'bar'
            return lang_cls
        return before_creation

    @registry.callbacks(f'{name}_after')
    def make_after_creation():

        def after_creation(nlp):
            nonlocal ran_after
            ran_after = True
            assert isinstance(nlp, English)
            assert nlp.pipe_names == []
            assert nlp.Defaults.foo == 'bar'
            nlp.meta['foo'] = 'bar'
            return nlp
        return after_creation

    @registry.callbacks(f'{name}_after_pipeline')
    def make_after_pipeline_creation():

        def after_pipeline_creation(nlp):
            nonlocal ran_after_pipeline
            ran_after_pipeline = True
            assert isinstance(nlp, English)
            assert nlp.pipe_names == ['sentencizer']
            assert nlp.Defaults.foo == 'bar'
            assert nlp.meta['foo'] == 'bar'
            nlp.meta['bar'] = 'baz'
            return nlp
        return after_pipeline_creation

    @registry.callbacks(f'{name}_before_init')
    def make_before_init():

        def before_init(nlp):
            nonlocal ran_before_init
            ran_before_init = True
            nlp.meta['before_init'] = 'before'
            return nlp
        return before_init

    @registry.callbacks(f'{name}_after_init')
    def make_after_init():

        def after_init(nlp):
            nonlocal ran_after_init
            ran_after_init = True
            nlp.meta['after_init'] = 'after'
            return nlp
        return after_init
    config = {'nlp': {'pipeline': ['sentencizer'], 'before_creation': {'@callbacks': f'{name}_before'}, 'after_creation': {'@callbacks': f'{name}_after'}, 'after_pipeline_creation': {'@callbacks': f'{name}_after_pipeline'}}, 'components': {'sentencizer': {'factory': 'sentencizer'}}, 'initialize': {'before_init': {'@callbacks': f'{name}_before_init'}, 'after_init': {'@callbacks': f'{name}_after_init'}}}
    nlp = English.from_config(config)
    assert nlp.Defaults.foo == 'bar'
    assert nlp.meta['foo'] == 'bar'
    assert nlp.meta['bar'] == 'baz'
    assert 'before_init' not in nlp.meta
    assert 'after_init' not in nlp.meta
    assert nlp.pipe_names == ['sentencizer']
    assert nlp('text')
    nlp.initialize()
    assert nlp.meta['before_init'] == 'before'
    assert nlp.meta['after_init'] == 'after'
    assert all([ran_before, ran_after, ran_after_pipeline, ran_before_init, ran_after_init])