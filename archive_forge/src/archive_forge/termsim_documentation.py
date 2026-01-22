from array import array
from itertools import chain
import logging
from math import sqrt
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus
Get the inner product(s) between real vectors / corpora X and Y.

        Return the inner product(s) between real vectors / corpora vec1 and vec2 expressed in a
        non-orthogonal normalized basis, where the dot product between the basis vectors is given by
        the sparse term similarity matrix.

        Parameters
        ----------
        vec1 : list of (int, float) or iterable of list of (int, float)
            A query vector / corpus in the sparse bag-of-words format.
        vec2 : list of (int, float) or iterable of list of (int, float)
            A document vector / corpus in the sparse bag-of-words format.
        normalized : tuple of {True, False, 'maintain'}, optional
            First/second value specifies whether the query/document vectors in the inner product
            will be L2-normalized (True; corresponds to the soft cosine measure), maintain their
            L2-norm during change of basis ('maintain'; corresponds to query expansion with partial
            membership), or kept as-is (False; corresponds to query expansion; default).

        Returns
        -------
        `self.matrix.dtype`,  `scipy.sparse.csr_matrix`, or :class:`numpy.matrix`
            The inner product(s) between `X` and `Y`.

        References
        ----------
        The soft cosine measure was perhaps first described by [sidorovetal14]_.
        Further notes on the efficient implementation of the soft cosine measure are described by
        [novotny18]_.

        .. [sidorovetal14] Grigori Sidorov et al., "Soft Similarity and Soft Cosine Measure: Similarity
           of Features in Vector Space Model", 2014, http://www.cys.cic.ipn.mx/ojs/index.php/CyS/article/view/2043/1921.

        .. [novotny18] Vít Novotný, "Implementation Notes for the Soft Cosine Measure", 2018,
           http://dx.doi.org/10.1145/3269206.3269317.

        