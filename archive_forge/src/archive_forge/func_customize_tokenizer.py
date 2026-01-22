import pickle
import re
import pytest
from spacy.attrs import ENT_IOB, ENT_TYPE
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import (
from ..util import assert_packed_msg_equal, make_tempdir
def customize_tokenizer(nlp):
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
    infix_re = compile_infix_regex(nlp.Defaults.infixes)
    exceptions = {k: v for k, v in dict(nlp.Defaults.tokenizer_exceptions).items() if not (len(k) == 2 and k[1] == '.')}
    new_tokenizer = Tokenizer(nlp.vocab, exceptions, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer, token_match=nlp.tokenizer.token_match, faster_heuristics=False)
    nlp.tokenizer = new_tokenizer