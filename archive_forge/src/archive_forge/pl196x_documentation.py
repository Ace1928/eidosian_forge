from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader

        In the pl196x corpus each category is stored in single
        file and thus both methods provide identical functionality. In order
        to accommodate finer granularity, a non-standard textids() method was
        implemented. All the main functions can be supplied with a list
        of required chunks---giving much more control to the user.
        