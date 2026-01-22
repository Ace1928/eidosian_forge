from __future__ import with_statement
import logging
from gensim import utils
from gensim.corpora import LowCorpus
Get the document stored in file by `offset` position.

        Parameters
        ----------
        offset : int
            Offset (in bytes) to begin of document.

        Returns
        -------
        list of (int, int)
            Document in BoW format (+"document_id" and "lang" if metadata=True).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.corpora import MalletCorpus
            >>>
            >>> data = MalletCorpus(datapath("testcorpus.mallet"))
            >>> data.docbyoffset(1)  # end of first line
            [(3, 1), (4, 1)]
            >>> data.docbyoffset(4)  # start of second line
            [(4, 1)]

        