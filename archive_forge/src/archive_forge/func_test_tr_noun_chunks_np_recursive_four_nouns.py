from spacy.tokens import Doc
def test_tr_noun_chunks_np_recursive_four_nouns(tr_tokenizer):
    text = 'kızına piyano dersi verdiğim hanım'
    heads = [3, 2, 3, 4, 4]
    deps = ['obl', 'nmod', 'obj', 'acl', 'ROOT']
    pos = ['NOUN', 'NOUN', 'NOUN', 'VERB', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'kızına piyano dersi verdiğim hanım '