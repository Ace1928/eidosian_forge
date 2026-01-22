from spacy.tokens import Doc
def test_tr_noun_chunks_nmod_simple(tr_tokenizer):
    text = 'arkadaşımın kedisi'
    heads = [1, 1]
    deps = ['nmod', 'ROOT']
    pos = ['NOUN', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'arkadaşımın kedisi '