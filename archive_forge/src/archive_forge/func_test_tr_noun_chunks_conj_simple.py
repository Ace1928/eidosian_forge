from spacy.tokens import Doc
def test_tr_noun_chunks_conj_simple(tr_tokenizer):
    text = 'Sen ya da ben'
    heads = [0, 3, 1, 0]
    deps = ['ROOT', 'cc', 'fixed', 'conj']
    pos = ['PRON', 'CCONJ', 'CCONJ', 'PRON']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text_with_ws == 'ben '
    assert chunks[1].text_with_ws == 'Sen '