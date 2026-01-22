import gc
import numpy
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.lang.en.syntax_iterators import noun_chunks
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList, get_arg_names, make_tempdir
from spacy.vocab import Vocab
def test_disable_enable_pipes():
    name = 'test_disable_enable_pipes'
    results = {}

    def make_component(name):
        results[name] = ''

        def component(doc):
            nonlocal results
            results[name] = doc.text
            return doc
        return component
    c1 = Language.component(f'{name}1', func=make_component(f'{name}1'))
    c2 = Language.component(f'{name}2', func=make_component(f'{name}2'))
    nlp = Language()
    nlp.add_pipe(f'{name}1')
    nlp.add_pipe(f'{name}2')
    assert results[f'{name}1'] == ''
    assert results[f'{name}2'] == ''
    assert nlp.pipeline == [(f'{name}1', c1), (f'{name}2', c2)]
    assert nlp.pipe_names == [f'{name}1', f'{name}2']
    nlp.disable_pipe(f'{name}1')
    assert nlp.disabled == [f'{name}1']
    assert nlp.component_names == [f'{name}1', f'{name}2']
    assert nlp.pipe_names == [f'{name}2']
    assert nlp.config['nlp']['disabled'] == [f'{name}1']
    nlp('hello')
    assert results[f'{name}1'] == ''
    assert results[f'{name}2'] == 'hello'
    nlp.enable_pipe(f'{name}1')
    assert nlp.disabled == []
    assert nlp.pipe_names == [f'{name}1', f'{name}2']
    assert nlp.config['nlp']['disabled'] == []
    nlp('world')
    assert results[f'{name}1'] == 'world'
    assert results[f'{name}2'] == 'world'
    nlp.disable_pipe(f'{name}2')
    nlp.remove_pipe(f'{name}2')
    assert nlp.components == [(f'{name}1', c1)]
    assert nlp.pipeline == [(f'{name}1', c1)]
    assert nlp.component_names == [f'{name}1']
    assert nlp.pipe_names == [f'{name}1']
    assert nlp.disabled == []
    assert nlp.config['nlp']['disabled'] == []
    nlp.rename_pipe(f'{name}1', name)
    assert nlp.components == [(name, c1)]
    assert nlp.component_names == [name]
    nlp('!')
    assert results[f'{name}1'] == '!'
    assert results[f'{name}2'] == 'world'
    with pytest.raises(ValueError):
        nlp.disable_pipe(f'{name}2')
    nlp.disable_pipe(name)
    assert nlp.component_names == [name]
    assert nlp.pipe_names == []
    assert nlp.config['nlp']['disabled'] == [name]
    nlp('?')
    assert results[f'{name}1'] == '!'