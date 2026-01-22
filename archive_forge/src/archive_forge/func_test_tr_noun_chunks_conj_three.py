from spacy.tokens import Doc
def test_tr_noun_chunks_conj_three(tr_tokenizer):
    text = 'sen, ben ve ondan'
    heads = [0, 2, 0, 4, 0]
    deps = ['ROOT', 'punct', 'conj', 'cc', 'conj']
    pos = ['PRON', 'PUNCT', 'PRON', 'CCONJ', 'PRON']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 3
    assert chunks[0].text_with_ws == 'ondan '
    assert chunks[1].text_with_ws == 'ben '
    assert chunks[2].text_with_ws == 'sen '