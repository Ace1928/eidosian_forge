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
def segment_and_write_all_articles(file_path, output_file, min_article_character=200, workers=None, include_interlinks=False):
    """Write article title and sections to `output_file` (or stdout, if output_file is None).

    The output format is one article per line, in json-line format with 4 fields::

        'title' - title of article,
        'section_titles' - list of titles of sections,
        'section_texts' - list of content from sections,
        (Optional) 'section_interlinks' - list of interlinks in the article.

    Parameters
    ----------
    file_path : str
        Path to MediaWiki dump, typical filename is <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
        or <LANG>wiki-latest-pages-articles.xml.bz2.

    output_file : str or None
        Path to output file in json-lines format, or None for printing to stdout.

    min_article_character : int, optional
        Minimal number of character for article (except titles and leading gaps).

    workers: int or None
        Number of parallel workers, max(1, multiprocessing.cpu_count() - 1) if None.

    include_interlinks: bool
        Whether or not interlinks should be included in the output
    """
    if output_file is None:
        outfile = getattr(sys.stdout, 'buffer', sys.stdout)
    else:
        outfile = gensim.utils.open(output_file, 'wb')
    try:
        article_stream = segment_all_articles(file_path, min_article_character, workers=workers, include_interlinks=include_interlinks)
        for idx, article in enumerate(article_stream):
            article_title, article_sections = (article[0], article[1])
            if include_interlinks:
                interlinks = article[2]
            output_data = {'title': article_title, 'section_titles': [], 'section_texts': []}
            if include_interlinks:
                output_data['interlinks'] = interlinks
            for section_heading, section_content in article_sections:
                output_data['section_titles'].append(section_heading)
                output_data['section_texts'].append(section_content)
            if (idx + 1) % 100000 == 0:
                logger.info('processed #%d articles (at %r now)', idx + 1, article_title)
            outfile.write((json.dumps(output_data) + '\n').encode('utf-8'))
    finally:
        if output_file is not None:
            outfile.close()