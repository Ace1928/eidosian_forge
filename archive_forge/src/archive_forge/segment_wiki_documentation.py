import argparse
import json
import logging
import multiprocessing
import re
import sys
from xml.etree import ElementTree
from functools import partial
from gensim.corpora.wikicorpus import IGNORED_NAMESPACES, WikiCorpus, filter_wiki, find_interlinks, get_namespace, utils
import gensim.utils
Iterate over the dump, returning titles and text versions of all sections of articles.

        Notes
        -----
        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).

        Note that this iterates over the **texts**; if you want vectors, just use
        the standard corpus interface instead of this function:

        .. sourcecode:: pycon

            >>> for vec in wiki_corpus:
            >>>     print(vec)

        Yields
        ------
        (str, list of (str, str), list of (str, str))
            Structure contains (title, [(section_heading, section_content), ...],
            (Optionally)[(interlink_article, interlink_text), ...]).

        