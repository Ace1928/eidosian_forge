from spacy.compat import pickle
from spacy.language import Language
def test_list_of_docs_pickles_efficiently():
    nlp = Language()
    for i in range(10000):
        _ = nlp.vocab[str(i)]
    one_pickled = pickle.dumps(nlp('0'), -1)
    docs = list(nlp.pipe((str(i) for i in range(100))))
    many_pickled = pickle.dumps(docs, -1)
    assert len(many_pickled) < len(one_pickled) * 2
    many_unpickled = pickle.loads(many_pickled)
    assert many_unpickled[0].text == '0'
    assert many_unpickled[-1].text == '99'
    assert len(many_unpickled) == 100