import os
from .regexp_tokenizer import RegexpTokenizer
from .simple_tokenizer import SimpleTokenizer
from .corenlp_tokenizer import CoreNLPTokenizer  # noqa: E402
def get_annotators_for_model(model):
    return get_annotators_for_args(model.args)