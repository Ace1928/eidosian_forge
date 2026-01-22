import re
from wasabi import Printer
from ...tokens import Doc, Span, Token
from ...training import biluo_tags_to_spans, iob_to_biluo
from ...vocab import Vocab
from .conll_ner_to_docs import n_sents_info
def conllu_to_docs(input_data, n_sents=10, append_morphology=False, ner_map=None, merge_subtokens=False, no_print=False, **_):
    """
    Convert conllu files into JSON format for use with train cli.
    append_morphology parameter enables appending morphology to tags, which is
    useful for languages such as Spanish, where UD tags are not so rich.

    Extract NER tags if available and convert them so that they follow
    BILUO and the Wikipedia scheme
    """
    MISC_NER_PATTERN = '^((?:name|NE)=)?([BILU])-([A-Z_]+)|O$'
    msg = Printer(no_print=no_print)
    n_sents_info(msg, n_sents)
    sent_docs = read_conllx(input_data, append_morphology=append_morphology, ner_tag_pattern=MISC_NER_PATTERN, ner_map=ner_map, merge_subtokens=merge_subtokens)
    sent_docs_to_merge = []
    for sent_doc in sent_docs:
        sent_docs_to_merge.append(sent_doc)
        if len(sent_docs_to_merge) % n_sents == 0:
            yield Doc.from_docs(sent_docs_to_merge)
            sent_docs_to_merge = []
    if sent_docs_to_merge:
        yield Doc.from_docs(sent_docs_to_merge)