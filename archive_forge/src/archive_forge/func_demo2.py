from math import log
from operator import itemgetter
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI
def demo2():
    from nltk.corpus import treebank
    d = list(treebank.tagged_sents())
    t = TnT(N=1000, C=False)
    s = TnT(N=1000, C=True)
    t.train(d[11 * 100:])
    s.train(d[11 * 100:])
    for i in range(10):
        tacc = t.accuracy(d[i * 100:(i + 1) * 100])
        tp_un = t.unknown / (t.known + t.unknown)
        tp_kn = t.known / (t.known + t.unknown)
        t.unknown = 0
        t.known = 0
        print('Capitalization off:')
        print('Accuracy:', tacc)
        print('Percentage known:', tp_kn)
        print('Percentage unknown:', tp_un)
        print('Accuracy over known words:', tacc / tp_kn)
        sacc = s.accuracy(d[i * 100:(i + 1) * 100])
        sp_un = s.unknown / (s.known + s.unknown)
        sp_kn = s.known / (s.known + s.unknown)
        s.unknown = 0
        s.known = 0
        print('Capitalization on:')
        print('Accuracy:', sacc)
        print('Percentage known:', sp_kn)
        print('Percentage unknown:', sp_un)
        print('Accuracy over known words:', sacc / sp_kn)