from spacy.tokens import Doc
def test_en_parser_noun_chunks_pp_chunks(en_vocab):
    words = ['A', 'phrase', 'with', 'another', 'phrase', 'occurs', '.']
    heads = [1, 5, 1, 4, 2, 5, 5]
    pos = ['DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'VERB', 'PUNCT']
    deps = ['det', 'nsubj', 'prep', 'det', 'pobj', 'ROOT', 'punct']
    doc = Doc(en_vocab, words=words, pos=pos, deps=deps, heads=heads)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text_with_ws == 'A phrase '
    assert chunks[1].text_with_ws == 'another phrase '