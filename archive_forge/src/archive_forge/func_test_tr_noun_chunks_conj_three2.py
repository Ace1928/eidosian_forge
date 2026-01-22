from spacy.tokens import Doc
def test_tr_noun_chunks_conj_three2(tr_tokenizer):
    text = 'ben ya da sen ya da onlar'
    heads = [0, 3, 1, 0, 6, 4, 3]
    deps = ['ROOT', 'cc', 'fixed', 'conj', 'cc', 'fixed', 'conj']
    pos = ['PRON', 'CCONJ', 'CCONJ', 'PRON', 'CCONJ', 'CCONJ', 'PRON']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 3
    assert chunks[0].text_with_ws == 'onlar '
    assert chunks[1].text_with_ws == 'sen '
    assert chunks[2].text_with_ws == 'ben '