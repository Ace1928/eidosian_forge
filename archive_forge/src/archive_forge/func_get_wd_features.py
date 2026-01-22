import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def get_wd_features(dict, hypothesis, history, wd_features, wd_weights):
    """
    Given a conversational history and a hypothesis (i.e. partially generated response),
    compute the Weighted Decoding features for all words in the vocabulary.

    Inputs:
        dict: parlai DictionaryAgent
        hypothesis: list of ints or None
        history: a ConvAI2History. This represents the conversation history.
        wd_features: list of strings; the names of the WD features we want to use
        wd_weights: list of floats; the weights corresponding to the WD features.
    Returns:
        wd_feat_vec: tensor shape (vocab_size), containing weighted sum of the feature
          functions, for each candidate continuation word
    """
    wd_feat_vec = torch.zeros(len(dict))
    for f, w in zip(wd_features, wd_weights):
        wd_feat_vec = WDFEATURE2UPDATEFN[f]((dict, hypothesis, history, w, wd_feat_vec))
    return wd_feat_vec