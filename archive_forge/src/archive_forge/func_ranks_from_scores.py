def ranks_from_scores(scores, rank_gap=1e-15):
    """Given a sequence of (key, score) tuples, yields each key with an
    increasing rank, tying with previous key's rank if the difference between
    their scores is less than rank_gap. Suitable for use as an argument to
    ``spearman_correlation``.
    """
    prev_score = None
    rank = 0
    for i, (key, score) in enumerate(scores):
        try:
            if abs(score - prev_score) > rank_gap:
                rank = i
        except TypeError:
            pass
        yield (key, rank)
        prev_score = score