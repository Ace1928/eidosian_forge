import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def lastutt_sim_arora_sent(utt, history):
    """
    Sentence-level attribute function. See explanation above.

    Returns
      cos_sim(sent_emb(last_utt), sent_emb(utt))
    the cosine similarity of the Arora-style sentence embeddings for the current
    response (utt) and the partner's last utterance (last_utt, which is in history).

    - If there is no last_utt (i.e. utt is the first utterance of the conversation),
      returns None.
    - If one or both of utt and last_utt are all-OOV; thus we can't compute sentence
      embeddings, return the string 'oov'.
    """
    partner_utts = history.partner_utts
    if len(partner_utts) == 0:
        return None
    last_utt = partner_utts[-1]
    if '__SILENCE__' in last_utt:
        assert last_utt.strip() == '__SILENCE__'
        return None
    last_utt_emb = sent_embedder.embed_sent(last_utt.split())
    response_emb = sent_embedder.embed_sent(utt.split())
    if last_utt_emb is None or response_emb is None:
        return 'oov'
    sim = torch.nn.functional.cosine_similarity(last_utt_emb, response_emb, dim=0)
    return sim.item()