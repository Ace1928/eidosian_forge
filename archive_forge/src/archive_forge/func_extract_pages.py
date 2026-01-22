import bz2
import logging
import multiprocessing
import re
import signal
from pickle import PicklingError
from xml.etree.ElementTree import iterparse
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
def extract_pages(f, filter_namespaces=False, filter_articles=None):
    """Extract pages from a MediaWiki database dump.

    Parameters
    ----------
    f : file
        File-like object.
    filter_namespaces : list of str or bool
         Namespaces that will be extracted.

    Yields
    ------
    tuple of (str or None, str, str)
        Title, text and page id.

    """
    elems = (elem for _, elem in iterparse(f, events=('end',)))
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {'ns': namespace}
    page_tag = '{%(ns)s}page' % ns_mapping
    text_path = './{%(ns)s}revision/{%(ns)s}text' % ns_mapping
    title_path = './{%(ns)s}title' % ns_mapping
    ns_path = './{%(ns)s}ns' % ns_mapping
    pageid_path = './{%(ns)s}id' % ns_mapping
    for elem in elems:
        if elem.tag == page_tag:
            title = elem.find(title_path).text
            text = elem.find(text_path).text
            if filter_namespaces:
                ns = elem.find(ns_path).text
                if ns not in filter_namespaces:
                    text = None
            if filter_articles is not None:
                if not filter_articles(elem, namespace=namespace, title=title, text=text, page_tag=page_tag, text_path=text_path, title_path=title_path, ns_path=ns_path, pageid_path=pageid_path):
                    text = None
            pageid = elem.find(pageid_path).text
            yield (title, text or '', pageid)
            elem.clear()