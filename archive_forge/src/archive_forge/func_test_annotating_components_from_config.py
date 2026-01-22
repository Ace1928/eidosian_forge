from typing import Callable, Iterable, Iterator
import pytest
from thinc.api import Config
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
from spacy.training.loop import train
from spacy.util import load_model_from_config, registry
def test_annotating_components_from_config(config_str):

    @registry.readers('unannotated_corpus')
    def create_unannotated_corpus() -> Callable[[Language], Iterable[Example]]:
        return UnannotatedCorpus()

    class UnannotatedCorpus:

        def __call__(self, nlp: Language) -> Iterator[Example]:
            for text in ['a a', 'b b', 'c c']:
                doc = nlp.make_doc(text)
                yield Example(doc, doc)
    orig_config = Config().from_str(config_str)
    nlp = load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.config['training']['annotating_components'] == ['sentencizer']
    train(nlp)
    nlp.config['training']['annotating_components'] = []
    with pytest.raises(ValueError):
        train(nlp)