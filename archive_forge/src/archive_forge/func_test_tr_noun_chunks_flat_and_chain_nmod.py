from spacy.tokens import Doc
def test_tr_noun_chunks_flat_and_chain_nmod(tr_tokenizer):
    text = 'Batı Afrika ülkelerinden Sierra Leone'
    heads = [1, 2, 3, 3, 3]
    deps = ['nmod', 'nmod', 'nmod', 'ROOT', 'flat']
    pos = ['NOUN', 'PROPN', 'NOUN', 'PROPN', 'PROPN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'Batı Afrika ülkelerinden Sierra Leone '