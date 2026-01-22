import pytest
from numpy.testing import assert_equal
from thinc.api import Adam
from spacy import registry, util
from spacy.attrs import DEP, NORM
from spacy.lang.en import English
from spacy.pipeline import DependencyParser
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
from ..util import apply_transition_sequence, make_tempdir
@pytest.mark.parametrize('pipe_name', ['parser', 'beam_parser'])
def test_overfitting_IO(pipe_name):
    nlp = English()
    parser = nlp.add_pipe(pipe_name)
    train_examples = []
    for text, annotations in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for dep in annotations.get('deps', []):
            parser.add_label(dep)
    optimizer = nlp.initialize()
    for i in range(200):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses[pipe_name] < 0.0001
    test_text = 'I like securities.'
    doc = nlp(test_text)
    assert doc[0].dep_ == 'nsubj'
    assert doc[2].dep_ == 'dobj'
    assert doc[3].dep_ == 'punct'
    assert doc[0].head.i == 1
    assert doc[2].head.i == 1
    assert doc[3].head.i == 1
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        assert doc2[0].dep_ == 'nsubj'
        assert doc2[2].dep_ == 'dobj'
        assert doc2[3].dep_ == 'punct'
        assert doc2[0].head.i == 1
        assert doc2[2].head.i == 1
        assert doc2[3].head.i == 1
    texts = ['Just a sentence.', 'Then one more sentence about London.', 'Here is another one.', 'I like London.']
    batch_deps_1 = [doc.to_array([DEP]) for doc in nlp.pipe(texts)]
    batch_deps_2 = [doc.to_array([DEP]) for doc in nlp.pipe(texts)]
    no_batch_deps = [doc.to_array([DEP]) for doc in [nlp(text) for text in texts]]
    assert_equal(batch_deps_1, batch_deps_2)
    assert_equal(batch_deps_1, no_batch_deps)