from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader, XMLCorpusView
def handle_sent(self, elt):
    sent = []
    for child in elt:
        if child.tag in ('mw', 'hi', 'corr', 'trunc'):
            sent += [self.handle_word(w) for w in child]
        elif child.tag in ('w', 'c'):
            sent.append(self.handle_word(child))
        elif child.tag not in self.tags_to_ignore:
            raise ValueError('Unexpected element %s' % child.tag)
    return BNCSentence(elt.attrib['n'], sent)