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
def write_sample_training(tmp_dir):
    words = ['The', 'players', 'start', '.']
    tags = ['DT', 'NN', 'VBZ', '.']
    doc = Doc(English().vocab, words=words, tags=tags)
    doc_bin = DocBin()
    doc_bin.add(doc)
    train_path = f'{tmp_dir}/train.spacy'
    dev_path = f'{tmp_dir}/dev.spacy'
    doc_bin.to_disk(train_path)
    doc_bin.to_disk(dev_path)
    return (train_path, dev_path)