import pytest
from thinc.api import Adam, fix_random_seed
from spacy import registry
from spacy.attrs import NORM
from spacy.language import Language
from spacy.pipeline import DependencyParser, EntityRecognizer
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_add_label_deserializes_correctly():
    cfg = {'model': DEFAULT_NER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    ner1 = EntityRecognizer(Vocab(), model)
    ner1.add_label('C')
    ner1.add_label('B')
    ner1.add_label('A')
    ner1.initialize(lambda: [_ner_example(ner1)])
    ner2 = EntityRecognizer(Vocab(), model)
    ner2.model.attrs['resize_output'](ner2.model, ner1.moves.n_moves)
    ner2.from_bytes(ner1.to_bytes())
    assert ner1.moves.n_moves == ner2.moves.n_moves
    for i in range(ner1.moves.n_moves):
        assert ner1.moves.get_class_name(i) == ner2.moves.get_class_name(i)