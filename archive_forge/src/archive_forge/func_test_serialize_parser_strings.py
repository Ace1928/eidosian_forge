import pickle
import pytest
import srsly
from thinc.api import Linear
import spacy
from spacy import Vocab, load, registry
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import (
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.tokens import Span
from spacy.util import ensure_path, load_model
from ..util import make_tempdir
@pytest.mark.parametrize('Parser', test_parsers)
def test_serialize_parser_strings(Parser):
    vocab1 = Vocab()
    label = 'FunnyLabel'
    assert label not in vocab1.strings
    cfg = {'model': DEFAULT_PARSER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    parser1 = Parser(vocab1, model)
    parser1.add_label(label)
    assert label in parser1.vocab.strings
    vocab2 = Vocab()
    assert label not in vocab2.strings
    parser2 = Parser(vocab2, model)
    parser2 = parser2.from_bytes(parser1.to_bytes(exclude=['vocab']))
    assert label in parser2.vocab.strings