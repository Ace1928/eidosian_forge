import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
def create_spacyLanguage(config, bytes):
    lang_cls = spacy.util.get_lang_class(config['nlp']['lang'])
    lang_inst = lang_cls.from_config(config)
    return lang_inst.from_bytes(bytes)