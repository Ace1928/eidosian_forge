from spacy.tokens import Doc
def test_tr_noun_chunks_acl_nmod(tr_tokenizer):
    text = 'en sevdiğim ses sanatçısı'
    heads = [1, 3, 3, 3]
    deps = ['advmod', 'acl', 'nmod', 'ROOT']
    pos = ['ADV', 'VERB', 'NOUN', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'en sevdiğim ses sanatçısı '