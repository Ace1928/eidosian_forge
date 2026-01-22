from spacy.tokens import Doc
def test_tr_noun_chunks_np_recursive_no_nmod(tr_tokenizer):
    text = 'içine birkaç çiçek konmuş olan bir vazo'
    heads = [3, 2, 3, 6, 3, 6, 6]
    deps = ['obl', 'det', 'nsubj', 'acl', 'aux', 'det', 'ROOT']
    pos = ['ADP', 'DET', 'NOUN', 'VERB', 'AUX', 'DET', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'içine birkaç çiçek konmuş olan bir vazo '