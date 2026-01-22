import math
from collections.abc import Sequence
import heapq
import json
import torch
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
def rank_candidates(query_rep, cands, length_penalty, dictionary=None):
    """
    Rank candidates given representation of query.

    :param query_rep: base query representation to match text again.
    :param cands: strings to compare against query_rep for matching tokens
    :param length_penalty: scores are divided by the norm taken to this power
    :dictionary: optional dictionary to use to tokenize text

    :returns: ordered list of candidate strings in score-ranked order
    """
    if True:
        mpq = MaxPriorityQueue(100)
        for c in cands:
            score = score_match(query_rep, c, length_penalty, dictionary)
            mpq.add(c, score)
        return list(reversed(mpq))
    else:
        cands = list(cands)
        score = [0] * len(cands)
        for i, c in enumerate(cands):
            score[i] = -score_match(query_rep, c, length_penalty, dictionary)
        r = [i[0] for i in sorted(enumerate(score), key=lambda x: x[1])]
        res = []
        for i in range(min(100, len(score))):
            res.append(cands[r[i]])
        return res