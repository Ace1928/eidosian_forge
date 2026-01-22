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
def extract_page_xmls(f):
    """Extract pages from a MediaWiki database dump.

    Parameters
    ----------
    f : file
        File descriptor of MediaWiki dump.

    Yields
    ------
    str
        XML strings for page tags.

    """
    elems = (elem for _, elem in ElementTree.iterparse(f, events=('end',)))
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {'ns': namespace}
    page_tag = '{%(ns)s}page' % ns_mapping
    for elem in elems:
        if elem.tag == page_tag:
            yield ElementTree.tostring(elem)
            elem.clear()