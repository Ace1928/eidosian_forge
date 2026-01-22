import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def lastutt_sim_arora_word(dict, hypothesis, history, wt, feat):
    """
    Weighted decoding feature function.

    See explanation above. Given a word w, this feature is equal to cos_sim(word_emb(w),
    sent_emb(l)) the cosine similarity between the GloVe vector for word w, and the
    Arora-style sentence embedding for the partner's last utterance l.
    """
    partner_utts = history.partner_utts
    if len(partner_utts) == 0:
        return feat
    last_utt = partner_utts[-1]
    if last_utt.strip().lower() == '__silence__':
        return feat
    last_utt_emb = sent_embedder.embed_sent(dict.tokenize(last_utt))
    if last_utt_emb is None:
        return feat
    sims = sent_embedder.get_word_sims(last_utt, last_utt_emb, dict)
    feat += wt * sims
    return feat