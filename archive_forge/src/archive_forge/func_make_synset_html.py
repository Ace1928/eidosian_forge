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
def make_synset_html(db_name, disp_name, rels):
    synset_html = '<i>%s</i>\n' % make_lookup_link(copy.deepcopy(ref).toggle_synset_relation(synset, db_name), disp_name)
    if db_name in ref.synset_relations[synset.name()]:
        synset_html += '<ul>%s</ul>\n' % ''.join(('<li>%s</li>\n' % relation_html(r) for r in rels))
    return synset_html