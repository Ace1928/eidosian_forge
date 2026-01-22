from spacy.tokens import Doc
def test_tr_noun_chunks_flat_name_lastname_and_title(tr_tokenizer):
    text = 'Cumhurbaşkanı Ahmet Necdet Sezer'
    heads = [1, 1, 1, 1]
    deps = ['nmod', 'ROOT', 'flat', 'flat']
    pos = ['NOUN', 'PROPN', 'PROPN', 'PROPN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'Cumhurbaşkanı Ahmet Necdet Sezer '