import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.hashdictionary import HashDictionary
from gensim.topic_coherence import probability_estimation
def build_segmented_topics(self):
    token2id = self.dictionary.token2id
    computer_id = token2id['computer']
    system_id = token2id['system']
    user_id = token2id['user']
    graph_id = token2id['graph']
    self.segmented_topics = [[(system_id, graph_id), (computer_id, graph_id), (computer_id, system_id)], [(computer_id, graph_id), (user_id, graph_id), (user_id, computer_id)]]
    self.computer_id = computer_id
    self.system_id = system_id
    self.user_id = user_id
    self.graph_id = graph_id