from spacy.tokens import Doc
def test_tr_noun_chunks_conj_noun_head_verb(tr_tokenizer):
    text = 'Simge babasını görmüyormuş, annesini değil'
    heads = [2, 2, 2, 4, 2, 4]
    deps = ['nsubj', 'obj', 'ROOT', 'punct', 'conj', 'aux']
    pos = ['PROPN', 'NOUN', 'VERB', 'PUNCT', 'NOUN', 'AUX']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 3
    assert chunks[0].text_with_ws == 'annesini '
    assert chunks[1].text_with_ws == 'babasını '
    assert chunks[2].text_with_ws == 'Simge '