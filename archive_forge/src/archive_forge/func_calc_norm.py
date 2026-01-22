import logging
from gensim import interfaces, matutils
def calc_norm(self, corpus):
    """Calculate the norm by calling :func:`~gensim.matutils.unitvec` with the norm parameter.

        Parameters
        ----------
        corpus : iterable of iterable of (int, number)
            Input corpus.

        """
    logger.info('Performing %s normalization...', self.norm)
    norms = []
    numnnz = 0
    docno = 0
    for bow in corpus:
        docno += 1
        numnnz += len(bow)
        norms.append(matutils.unitvec(bow, self.norm))
    self.num_docs = docno
    self.num_nnz = numnnz
    self.norms = norms