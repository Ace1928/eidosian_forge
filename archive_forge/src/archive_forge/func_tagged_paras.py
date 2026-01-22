import functools
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView, concat
@_parse_args
def tagged_paras(self, fileids=None, **kwargs):
    return concat([self._view(fileid, mode=IPIPANCorpusView.PARAS_MODE, **kwargs) for fileid in self._list_morph_files(fileids)])