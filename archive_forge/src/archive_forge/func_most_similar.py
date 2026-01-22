import logging
from gensim.similarities.termsim import TermSimilarityIndex
from gensim import utils
def most_similar(self, t1, topn=10):
    """kNN fuzzy search: find the `topn` most similar terms from `self.dictionary` to `t1`."""
    result = {}
    if self.max_distance > 0:
        effective_topn = topn + 1 if t1 in self.dictionary.token2id else topn
        effective_topn = min(len(self.dictionary), effective_topn)
        for distance in range(1, self.max_distance + 1):
            for t2 in self.index.query(t1, distance).get(distance, []):
                if t1 == t2:
                    continue
                similarity = self.levsim(t1, t2, distance)
                if similarity > 0:
                    result[t2] = similarity
            if len(result) >= effective_topn:
                break
    return sorted(result.items(), key=lambda x: (-x[1], x[0]))[:topn]