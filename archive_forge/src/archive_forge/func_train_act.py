from parlai.core.agents import Agent
from parlai.utils.misc import AttrDict
from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .build_tfidf import run as build_tfidf
from .build_tfidf import live_count_matrix, get_tfidf_matrix
from numpy.random import choice
from collections import deque
import math
import random
import os
import json
import sqlite3
def train_act(self):
    if 'ordered' not in self.opt.get('datatype', 'train:ordered') or self.opt.get('batchsize', 1) != 1 or self.opt.get('num_epochs', 1) != 1:
        raise RuntimeError('Need to set --batchsize 1, --datatype train:ordered, --num_epochs 1')
    obs = self.observation
    self.current.append(obs)
    self.episode_done = obs.get('episode_done', False)
    if self.episode_done:
        for ex in self.current:
            if 'text' in ex:
                text = ex['text']
                self.context.append(text)
                if len(self.context) > 1:
                    text = '\n'.join(self.context)
            labels = ex.get('labels', ex.get('eval_labels'))
            label = None
            if labels is not None:
                label = random.choice(labels)
                if self.include_labels:
                    self.context.append(label)
            self.triples_to_add.append((None, text, label))
        self.episode_done = False
        self.current.clear()
        self.context.clear()
    return {'id': self.getID(), 'text': obs.get('labels', ["I don't know"])[0]}