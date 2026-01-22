import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView

        :param fileids: A list specifying the fileids that should be used.
        :param tagset: The tagset that should be used in the returned object,
                       either "universal" or "msd", "msd" is the default
        :param tags: An MSD Tag that is used to filter all parts of the used corpus
                     that are not more precise or at least equal to the given tag
        :return: the given file(s) as a list of paragraphs, each encoded as a
                 list of sentences, which are in turn encoded as a list
                 of (word,tag) tuples
        :rtype: list(list(list(tuple(str, str))))
        