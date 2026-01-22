from wasabi import Printer
from ...errors import Errors
from ...tokens import Doc, Span
from ...training import iob_to_biluo
from ...util import get_lang_class, load_model
from .. import tags_to_entities
def segment_sents_and_docs(doc, n_sents, doc_delimiter, model=None, msg=None):
    sentencizer = None
    if model:
        nlp = load_model(model)
        if 'parser' in nlp.pipe_names:
            msg.info(f"Segmenting sentences with parser from model '{model}'.")
            for name, proc in nlp.pipeline:
                if 'parser' in getattr(proc, 'listening_components', []):
                    nlp.replace_listeners(name, 'parser', ['model.tok2vec'])
            sentencizer = nlp.get_pipe('parser')
    if not sentencizer:
        msg.info('Segmenting sentences with sentencizer. (Use `-b model` for improved parser-based sentence segmentation.)')
        nlp = get_lang_class('xx')()
        sentencizer = nlp.create_pipe('sentencizer')
    lines = doc.strip().split('\n')
    words = [line.strip().split()[0] for line in lines]
    nlpdoc = Doc(nlp.vocab, words=words)
    sentencizer(nlpdoc)
    lines_with_segs = []
    sent_count = 0
    for i, token in enumerate(nlpdoc):
        if token.is_sent_start:
            if n_sents and sent_count % n_sents == 0:
                lines_with_segs.append(doc_delimiter)
            lines_with_segs.append('')
            sent_count += 1
        lines_with_segs.append(lines[i])
    return '\n'.join(lines_with_segs)