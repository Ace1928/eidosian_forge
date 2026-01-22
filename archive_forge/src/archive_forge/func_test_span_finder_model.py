import pytest
from thinc.api import Config
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.span_finder import span_finder_default_config
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import fix_random_seed, make_tempdir, registry
def test_span_finder_model():
    nlp = Language()
    docs = [nlp('This is an example.'), nlp('This is the second example.')]
    docs[0].spans[SPANS_KEY] = [docs[0][3:4]]
    docs[1].spans[SPANS_KEY] = [docs[1][3:5]]
    total_tokens = 0
    for doc in docs:
        total_tokens += len(doc)
    config = Config().from_str(span_finder_default_config).interpolate()
    model = registry.resolve(config)['model']
    model.initialize(X=docs)
    predictions = model.predict(docs)
    assert len(predictions) == total_tokens
    assert len(predictions[0]) == 2