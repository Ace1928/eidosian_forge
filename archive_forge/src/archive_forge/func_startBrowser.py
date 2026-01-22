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
def startBrowser(url, server_ready):

    def run():
        server_ready.wait()
        time.sleep(1)
        webbrowser.open(url, new=2, autoraise=1)
    t = threading.Thread(target=run)
    t.start()
    return t