import pytest
from spacy.lang.en import English
from spacy.training import Example
from thinc.api import Config
@pytest.mark.parametrize('parser_config', [{'@architectures': 'spacy-legacy.TransitionBasedParser.v1', 'state_type': 'parser', 'extra_state_tokens': False, 'hidden_width': 66, 'maxout_pieces': 2, 'use_upper': True, 'tok2vec': DEFAULT_TOK2VEC_MODEL}])
def test_parser(parser_config):
    pipe_config = {'model': parser_config}
    nlp = English()
    parser = nlp.add_pipe('parser', config=pipe_config)
    train_examples = []
    for text, annotations in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for dep in annotations.get('deps', []):
            if dep is not None:
                parser.add_label(dep)
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(150):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['parser'] < 0.0001