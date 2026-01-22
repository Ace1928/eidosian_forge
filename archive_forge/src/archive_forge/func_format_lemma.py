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
def format_lemma(w):
    w = w.replace('_', ' ')
    if w.lower() == word:
        return _bold(w)
    else:
        ref = Reference(w)
        return make_lookup_link(ref, w)