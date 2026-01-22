import logging
import itertools
import zlib
from gensim import utils
Save the debug token=>id mapping to a text file.

        Warnings
        --------
        Only makes sense when `debug=True`, for debugging.

        Parameters
        ----------
        fname : str
            Path to output file.

        Notes
        -----
        The format is:
        `id[TAB]document frequency of this id[TAB]tab-separated set of words in UTF8 that map to this id[NEWLINE]`.


        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import HashDictionary
            >>> from gensim.test.utils import get_tmpfile
            >>>
            >>> corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
            >>> data = HashDictionary(corpus)
            >>> data.save_as_text(get_tmpfile("dictionary_in_text_format"))

        