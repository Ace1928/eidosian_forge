from __future__ import with_statement
import logging
from gensim import utils
from gensim.corpora import IndexedCorpus
def docbyoffset(self, offset):
    """Get the document stored at file position `offset`.

        Parameters
        ----------
        offset : int
            Document's position.

        Returns
        -------
        tuple of (int, float)

        """
    with utils.open(self.fname, 'rb') as f:
        f.seek(offset)
        return self.line2doc(f.readline())[0]