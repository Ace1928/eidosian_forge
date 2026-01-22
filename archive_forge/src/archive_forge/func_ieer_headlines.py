import html
import re
from collections import defaultdict
def ieer_headlines():
    from nltk.corpus import ieer
    from nltk.tree import Tree
    print('IEER: First 20 Headlines')
    print('=' * 45)
    trees = [(doc.docno, doc.headline) for file in ieer.fileids() for doc in ieer.parsed_docs(file)]
    for tree in trees[:20]:
        print()
        print('%s:\n%s' % tree)