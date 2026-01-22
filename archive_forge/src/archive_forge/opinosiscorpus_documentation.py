import os
import re
from gensim.corpora import Dictionary
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS
Load the downloaded corpus.

        Parameters
        ----------
        path : string
            Path to the extracted zip file. If 'summaries-gold' is in a folder
            called 'opinosis', then the Path parameter would be 'opinosis',
            either relative to you current working directory or absolute.
        