from __future__ import with_statement
import logging
from gensim import utils
from gensim.corpora import IndexedCorpus
def line2doc(self, line):
    """Get a document from a single line in SVMlight format.
        This method inverse of :meth:`~gensim.corpora.svmlightcorpus.SvmLightCorpus.doc2line`.

        Parameters
        ----------
        line : str
            Line in SVMLight format.

        Returns
        -------
        (list of (int, float), str)
            Document in BoW format and target class label.

        """
    line = utils.to_unicode(line)
    line = line[:line.find('#')].strip()
    if not line:
        return None
    parts = line.split()
    if not parts:
        raise ValueError('invalid line format in %s' % self.fname)
    target, fields = (parts[0], [part.rsplit(':', 1) for part in parts[1:]])
    doc = [(int(p1) - 1, float(p2)) for p1, p2 in fields if p1 != 'qid']
    return (doc, target)