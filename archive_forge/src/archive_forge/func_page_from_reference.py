import base64
import copy
import getopt
import io
import os
import pickle
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from sys import argv
from urllib.parse import unquote_plus
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
def page_from_reference(href):
    """
    Returns a tuple of the HTML page built and the new current word

    :param href: The hypertext reference to be solved
    :type href: str
    :return: A tuple (page,word), where page is the new current HTML page
             to be sent to the browser and
             word is the new current word
    :rtype: A tuple (str,str)
    """
    word = href.word
    pos_forms = defaultdict(list)
    words = word.split(',')
    words = [w for w in [w.strip().lower().replace(' ', '_') for w in words] if w != '']
    if len(words) == 0:
        return ('', 'Please specify a word to search for.')
    for w in words:
        for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
            form = wn.morphy(w, pos)
            if form and form not in pos_forms[pos]:
                pos_forms[pos].append(form)
    body = ''
    for pos, pos_str, name in _pos_tuples():
        if pos in pos_forms:
            body += _hlev(3, name) + '\n'
            for w in pos_forms[pos]:
                try:
                    body += _collect_all_synsets(w, pos, href.synset_relations)
                except KeyError:
                    pass
    if not body:
        body = "The word or words '%s' were not found in the dictionary." % word
    return (body, word)