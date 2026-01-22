import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def wav(self, utterance, start=0, end=None):
    wave = import_from_stdlib('wave')
    w = wave.open(self.open(utterance + '.wav'), 'rb')
    if end is None:
        end = w.getnframes()
    w.readframes(start)
    frames = w.readframes(end - start)
    tf = tempfile.TemporaryFile()
    out = wave.open(tf, 'w')
    out.setparams(w.getparams())
    out.writeframes(frames)
    out.close()
    tf.seek(0)
    return tf.read()