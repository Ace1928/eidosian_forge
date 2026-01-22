import pickle
import re
import pytest
from spacy.attrs import ENT_IOB, ENT_TYPE
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import (
from ..util import assert_packed_msg_equal, make_tempdir
def test_serialize_tokenizer_roundtrip_disk(en_tokenizer):
    tokenizer = en_tokenizer
    with make_tempdir() as d:
        file_path = d / 'tokenizer'
        tokenizer.to_disk(file_path)
        tokenizer_d = en_tokenizer.from_disk(file_path)
        assert tokenizer.to_bytes() == tokenizer_d.to_bytes()