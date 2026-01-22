from pathlib import Path
import numpy as np
import pytest
import srsly
from thinc.api import Config, get_current_ops
from spacy import util
from spacy.lang.en import English
from spacy.language import DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_PRETRAIN_PATH
from spacy.ml.models.multi_task import create_pretrain_vectors
from spacy.tokens import Doc, DocBin
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from spacy.training.pretrain import pretrain
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.parametrize('objective', VECTOR_OBJECTIVES)
def test_pretraining_tok2vec_vectors_fail(objective):
    """Test that pretraining doesn't works with the vectors objective if there are no static vectors"""
    config = Config().from_str(pretrain_string_listener)
    config['pretraining']['objective'] = objective
    nlp = util.load_model_from_config(config, auto_fill=True, validate=False)
    filled = nlp.config
    pretrain_config = util.load_config(DEFAULT_CONFIG_PRETRAIN_PATH)
    filled = pretrain_config.merge(filled)
    with make_tempdir() as tmp_dir:
        file_path = write_sample_jsonl(tmp_dir)
        filled['paths']['raw_text'] = file_path
        filled = filled.interpolate()
        assert filled['initialize']['vectors'] is None
        with pytest.raises(ValueError):
            pretrain(filled, tmp_dir)