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
def test_pretrain_default_vectors():
    nlp = English()
    nlp.add_pipe('tok2vec')
    nlp.initialize()
    nlp.vocab.vectors = Vectors(shape=(10, 10))
    create_pretrain_vectors(1, 1, 'cosine')(nlp.vocab, nlp.get_pipe('tok2vec').model)
    nlp.vocab.vectors = Vectors(data=get_current_ops().xp.zeros((10, 10)), mode='floret', hash_count=1)
    create_pretrain_vectors(1, 1, 'cosine')(nlp.vocab, nlp.get_pipe('tok2vec').model)
    with pytest.raises(ValueError, match='E875'):
        nlp.vocab.vectors = Vectors()
        create_pretrain_vectors(1, 1, 'cosine')(nlp.vocab, nlp.get_pipe('tok2vec').model)