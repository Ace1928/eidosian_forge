import pytest
from catalogue import RegistryError
from thinc.api import Config, ConfigValidationError
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import DEFAULT_CONFIG, DEFAULT_CONFIG_PRETRAIN_PATH, Language
from spacy.ml.models import (
from spacy.schemas import ConfigSchema, ConfigSchemaPretrain
from spacy.training import Example
from spacy.util import (
from ..util import make_tempdir
@pytest.mark.filterwarnings('ignore:\\[W036')
def test_config_overrides_registered_functions():
    nlp = spacy.blank('en')
    nlp.add_pipe('attribute_ruler')
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp_re1 = spacy.load(d, config={'components': {'attribute_ruler': {'scorer': {'@scorers': 'spacy.tagger_scorer.v1'}}}})
        assert nlp_re1.config['components']['attribute_ruler']['scorer']['@scorers'] == 'spacy.tagger_scorer.v1'

        @registry.misc('test_some_other_key')
        def misc_some_other_key():
            return 'some_other_key'
        nlp_re2 = spacy.load(d, config={'components': {'attribute_ruler': {'scorer': {'@scorers': 'spacy.overlapping_labeled_spans_scorer.v1', 'spans_key': {'@misc': 'test_some_other_key'}}}}})
        assert nlp_re2.config['components']['attribute_ruler']['scorer']['spans_key'] == {'@misc': 'test_some_other_key'}
        example = Example.from_dict(nlp_re2.make_doc('a b c'), {})
        scores = nlp_re2.evaluate([example])
        assert 'spans_some_other_key_f' in scores