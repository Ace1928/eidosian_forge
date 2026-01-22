import re
from wasabi import Printer
from ...tokens import Doc, Span, Token
from ...training import biluo_tags_to_spans, iob_to_biluo
from ...vocab import Vocab
from .conll_ner_to_docs import n_sents_info
def read_conllx(input_data, append_morphology=False, merge_subtokens=False, ner_tag_pattern='', ner_map=None):
    """Yield docs, one for each sentence"""
    vocab = Vocab()
    set_ents = has_ner(input_data, ner_tag_pattern)
    for sent in input_data.strip().split('\n\n'):
        lines = sent.strip().split('\n')
        if lines:
            while lines[0].startswith('#'):
                lines.pop(0)
            doc = conllu_sentence_to_doc(vocab, lines, ner_tag_pattern, merge_subtokens=merge_subtokens, append_morphology=append_morphology, ner_map=ner_map, set_ents=set_ents)
            yield doc