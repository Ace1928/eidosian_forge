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
def get_static_welcome_message():
    """
    Get the static welcome page.
    """
    return '\n<h3>Search Help</h3>\n<ul><li>The display below the line is an example of the output the browser\nshows you when you enter a search word. The search word was <b>green</b>.</li>\n<li>The search result shows for different parts of speech the <b>synsets</b>\ni.e. different meanings for the word.</li>\n<li>All underlined texts are hypertext links. There are two types of links:\nword links and others. Clicking a word link carries out a search for the word\nin the Wordnet database.</li>\n<li>Clicking a link of the other type opens a display section of data attached\nto that link. Clicking that link a second time closes the section again.</li>\n<li>Clicking <u>S:</u> opens a section showing the relations for that synset.</li>\n<li>Clicking on a relation name opens a section that displays the associated\nsynsets.</li>\n<li>Type a search word in the <b>Next Word</b> field and start the search by the\n<b>Enter/Return</b> key or click the <b>Search</b> button.</li>\n</ul>\n'