from spacy.tokens import Doc
def test_tr_noun_chunks_np_recursive_nsubj_to_root(tr_tokenizer):
    text = "Simge'nin okuduğu kitap"
    heads = [1, 2, 2]
    deps = ['nsubj', 'acl', 'ROOT']
    pos = ['PROPN', 'VERB', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == "Simge'nin okuduğu kitap "