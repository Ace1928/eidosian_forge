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
def pg(word, body):
    """
    Return a HTML page of NLTK Browser format constructed from the
    word and body

    :param word: The word that the body corresponds to
    :type word: str
    :param body: The HTML body corresponding to the word
    :type body: str
    :return: a HTML page for the word-body combination
    :rtype: str
    """
    return html_header % word + body + html_trailer