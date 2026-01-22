import math
from itertools import islice
from nltk.util import choose, ngrams
def corpus_ribes(list_of_references, hypotheses, alpha=0.25, beta=0.1):
    """
    This function "calculates RIBES for a system output (hypothesis) with
    multiple references, and returns "best" score among multi-references and
    individual scores. The scores are corpus-wise, i.e., averaged by the number
    of sentences." (c.f. RIBES version 1.03.1 code).

    Different from BLEU's micro-average precision, RIBES calculates the
    macro-average precision by averaging the best RIBES score for each pair of
    hypothesis and its corresponding references

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...          'because', 'he', 'read', 'the', 'book']

    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> round(corpus_ribes(list_of_references, hypotheses),4)
    0.3597

    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param alpha: hyperparameter used as a prior for the unigram precision.
    :type alpha: float
    :param beta: hyperparameter used as a prior for the brevity penalty.
    :type beta: float
    :return: The best ribes score from one of the references.
    :rtype: float
    """
    corpus_best_ribes = 0.0
    for references, hypothesis in zip(list_of_references, hypotheses):
        corpus_best_ribes += sentence_ribes(references, hypothesis, alpha, beta)
    return corpus_best_ribes / len(hypotheses)