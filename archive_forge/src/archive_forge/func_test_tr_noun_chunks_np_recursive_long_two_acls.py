from spacy.tokens import Doc
def test_tr_noun_chunks_np_recursive_long_two_acls(tr_tokenizer):
    text = "içine Simge'nin bahçesinden toplanmış birkaç çiçeğin konmuş olduğu bir vazo"
    heads = [6, 2, 3, 5, 5, 6, 9, 6, 9, 9]
    deps = ['obl', 'nmod', 'obl', 'acl', 'det', 'nsubj', 'acl', 'aux', 'det', 'ROOT']
    pos = ['ADP', 'PROPN', 'NOUN', 'VERB', 'DET', 'NOUN', 'VERB', 'AUX', 'DET', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == "içine Simge'nin bahçesinden toplanmış birkaç çiçeğin konmuş olduğu bir vazo "