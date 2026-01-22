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
@pytest.mark.parametrize('pipe_cls,n_moves,model_config', [(DependencyParser, 5, DEFAULT_PARSER_MODEL), (EntityRecognizer, 4, DEFAULT_NER_MODEL)])
def test_add_label_get_label(pipe_cls, n_moves, model_config):
    """Test that added labels are returned correctly. This test was added to
    test for a bug in DependencyParser.labels that'd cause it to fail when
    splitting the move names.
    """
    labels = ['A', 'B', 'C']
    model = registry.resolve({'model': model_config}, validate=True)['model']
    pipe = pipe_cls(Vocab(), model)
    for label in labels:
        pipe.add_label(label)
    assert len(pipe.move_names) == len(labels) * n_moves
    pipe_labels = sorted(list(pipe.labels))
    assert pipe_labels == labels