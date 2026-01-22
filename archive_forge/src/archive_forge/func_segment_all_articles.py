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
def segment_all_articles(file_path, min_article_character=200, workers=None, include_interlinks=False):
    """Extract article titles and sections from a MediaWiki bz2 database dump.

    Parameters
    ----------
    file_path : str
        Path to MediaWiki dump, typical filename is <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
        or <LANG>wiki-latest-pages-articles.xml.bz2.

    min_article_character : int, optional
        Minimal number of character for article (except titles and leading gaps).

    workers: int or None
        Number of parallel workers, max(1, multiprocessing.cpu_count() - 1) if None.

    include_interlinks: bool
        Whether or not interlinks should be included in the output

    Yields
    ------
    (str, list of (str, str), (Optionally) list of (str, str))
        Structure contains (title, [(section_heading, section_content), ...],
        (Optionally) [(interlink_article, interlink_text), ...]).

    """
    with gensim.utils.open(file_path, 'rb') as xml_fileobj:
        wiki_sections_corpus = _WikiSectionsCorpus(xml_fileobj, min_article_character=min_article_character, processes=workers, include_interlinks=include_interlinks)
        wiki_sections_corpus.metadata = True
        wiki_sections_text = wiki_sections_corpus.get_texts_with_sections()
        for article in wiki_sections_text:
            yield article