import logging
from collections import defaultdict
from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus
from gensim.matutils import MmReader
from gensim.matutils import MmWriter
def create_dictionary(self):
    """Generate :class:`gensim.corpora.dictionary.Dictionary` directly from the corpus and vocabulary data.

        Return
        ------
        :class:`gensim.corpora.dictionary.Dictionary`
            Dictionary, based on corpus.

        Examples
        --------

        .. sourcecode:: pycon

            >>> from gensim.corpora.ucicorpus import UciCorpus
            >>> from gensim.test.utils import datapath
            >>> ucc = UciCorpus(datapath('testcorpus.uci'))
            >>> dictionary = ucc.create_dictionary()

        """
    dictionary = Dictionary()
    dictionary.dfs = defaultdict(int)
    dictionary.id2token = self.id2word
    dictionary.token2id = utils.revdict(self.id2word)
    dictionary.num_docs = self.num_docs
    dictionary.num_nnz = self.num_nnz
    for docno, doc in enumerate(self):
        if docno % 10000 == 0:
            logger.info('PROGRESS: processing document %i of %i', docno, self.num_docs)
        for word, count in doc:
            dictionary.dfs[word] += 1
            dictionary.num_pos += count
    return dictionary