from typing import List, Tuple
import pytest
from nltk.tokenize import (
@pytest.mark.parametrize('input_text,n_sents,n_splits,lang_vars', [('Subject: Some subject. Attachments: Some attachments', 2, 1), ('Subject: Some subject! Attachments: Some attachments', 2, 1), ('This is just a normal sentence, just like any other.', 1, 0)])
def punkt_debug_decisions(self, input_text, n_sents, n_splits, lang_vars=None):
    tokenizer = punkt.PunktSentenceTokenizer()
    if lang_vars != None:
        tokenizer._lang_vars = lang_vars
    assert len(tokenizer.tokenize(input_text)) == n_sents
    assert len(list(tokenizer.debug_decisions(input_text))) == n_splits