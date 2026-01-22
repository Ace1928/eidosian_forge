from spacy.tokens import Doc
def test_de_parser_noun_chunks_standard_de(de_vocab):
    words = ['Eine', 'Tasse', 'steht', 'auf', 'dem', 'Tisch', '.']
    heads = [1, 2, 2, 2, 5, 3, 2]
    pos = ['DET', 'NOUN', 'VERB', 'ADP', 'DET', 'NOUN', 'PUNCT']
    deps = ['nk', 'sb', 'ROOT', 'mo', 'nk', 'nk', 'punct']
    doc = Doc(de_vocab, words=words, pos=pos, deps=deps, heads=heads)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text_with_ws == 'Eine Tasse '
    assert chunks[1].text_with_ws == 'dem Tisch '