from spacy.tokens import Doc
def test_tr_noun_chunks_det_amod_nmod(tr_tokenizer):
    text = 'bazı eski oyun kuralları'
    heads = [3, 3, 3, 3]
    deps = ['det', 'nmod', 'nmod', 'ROOT']
    pos = ['DET', 'ADJ', 'NOUN', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'bazı eski oyun kuralları '