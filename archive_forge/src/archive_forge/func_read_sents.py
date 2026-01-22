import codecs
from nltk.sem import evaluate
def read_sents(filename, encoding='utf8'):
    with codecs.open(filename, 'r', encoding) as fp:
        sents = [l.rstrip() for l in fp]
    sents = [l for l in sents if len(l) > 0]
    sents = [l for l in sents if not l[0] == '#']
    return sents