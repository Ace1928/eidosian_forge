import subprocess
import sys
from nltk.internals import find_binary
def encoding_demo():
    import sys
    from nltk.classify.maxent import TadmEventMaxentFeatureEncoding
    tokens = [({'f0': 1, 'f1': 1, 'f3': 1}, 'A'), ({'f0': 1, 'f2': 1, 'f4': 1}, 'B'), ({'f0': 2, 'f2': 1, 'f3': 1, 'f4': 1}, 'A')]
    encoding = TadmEventMaxentFeatureEncoding.train(tokens)
    write_tadm_file(tokens, encoding, sys.stdout)
    print()
    for i in range(encoding.length()):
        print('%s --> %d' % (encoding.describe(i), i))
    print()