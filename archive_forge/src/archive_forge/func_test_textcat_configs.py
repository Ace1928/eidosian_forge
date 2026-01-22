import random
import numpy.random
import pytest
from numpy.testing import assert_almost_equal
from thinc.api import Config, compounding, fix_random_seed, get_current_ops
from wasabi import msg
import spacy
from spacy import util
from spacy.cli.evaluate import print_prf_per_type, print_textcats_auc_per_cat
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import TextCategorizer
from spacy.pipeline.textcat import (
from spacy.pipeline.textcat_multilabel import (
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from spacy.training.initialize import init_nlp
from ..tok2vec import build_lazy_init_tok2vec as _
from ..util import make_tempdir
@pytest.mark.slow
@pytest.mark.parametrize('name,train_data,textcat_config', [('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatBOW.v1', 'exclusive_classes': False, 'ngram_size': 1, 'no_output_layer': False}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatBOW.v1', 'exclusive_classes': True, 'ngram_size': 4, 'no_output_layer': False}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatEnsemble.v1', 'exclusive_classes': False, 'pretrained_vectors': None, 'width': 64, 'embed_size': 2000, 'conv_depth': 2, 'window_size': 1, 'ngram_size': 1, 'dropout': None}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatEnsemble.v1', 'exclusive_classes': False, 'pretrained_vectors': None, 'width': 64, 'embed_size': 2000, 'conv_depth': 2, 'window_size': 1, 'ngram_size': 1, 'dropout': None}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatCNN.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': True}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatCNN.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': False}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatBOW.v2', 'exclusive_classes': False, 'ngram_size': 1, 'no_output_layer': False}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatBOW.v2', 'exclusive_classes': True, 'ngram_size': 4, 'no_output_layer': False}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatBOW.v2', 'exclusive_classes': False, 'ngram_size': 3, 'no_output_layer': True}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatBOW.v2', 'exclusive_classes': True, 'ngram_size': 2, 'no_output_layer': True}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': False, 'ngram_size': 1, 'no_output_layer': False}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': True, 'ngram_size': 4, 'no_output_layer': False}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': False, 'ngram_size': 3, 'no_output_layer': True}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': True, 'ngram_size': 2, 'no_output_layer': True}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatEnsemble.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'linear_model': {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': False, 'ngram_size': 1, 'no_output_layer': False}}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatEnsemble.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'linear_model': {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': True, 'ngram_size': 5, 'no_output_layer': False}}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatCNN.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': True}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatCNN.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': False}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatParametricAttention.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': True}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatParametricAttention.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': False}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy.TextCatReduce.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': True, 'use_reduce_first': True, 'use_reduce_last': True, 'use_reduce_max': True, 'use_reduce_mean': True}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy.TextCatReduce.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'exclusive_classes': False, 'use_reduce_first': True, 'use_reduce_last': True, 'use_reduce_max': True, 'use_reduce_mean': True})])
def test_textcat_configs(name, train_data, textcat_config):
    pipe_config = {'model': textcat_config}
    nlp = English()
    textcat = nlp.add_pipe(name, config=pipe_config)
    train_examples = []
    for text, annotations in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for label, value in annotations.get('cats').items():
            textcat.add_label(label)
    optimizer = nlp.initialize()
    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)