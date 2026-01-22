import math
import sys
import warnings
from collections import Counter
from fractions import Fraction
from nltk.util import ngrams
def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False):
    """
    Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all
    the hypotheses and their respective references.

    Instead of averaging the sentence level BLEU scores (i.e. macro-average
    precision), the original BLEU metric (Papineni et al. 2002) accounts for
    the micro-average precision (i.e. summing the numerators and denominators
    for each hypothesis-reference(s) pairs before the division).

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
    >>> corpus_bleu(list_of_references, hypotheses) # doctest: +ELLIPSIS
    0.5920...

    The example below show that corpus_bleu() is different from averaging
    sentence_bleu() for hypotheses

    >>> score1 = sentence_bleu([ref1a, ref1b, ref1c], hyp1)
    >>> score2 = sentence_bleu([ref2a], hyp2)
    >>> (score1 + score2) / 2 # doctest: +ELLIPSIS
    0.6223...

    Custom weights may be supplied to fine-tune the BLEU score further.
    A tuple of float weights for unigrams, bigrams, trigrams and so on can be given.
    >>> weights = (0.1, 0.3, 0.5, 0.1)
    >>> corpus_bleu(list_of_references, hypotheses, weights=weights) # doctest: +ELLIPSIS
    0.5818...

    This particular weight gave extra value to trigrams.
    Furthermore, multiple weights can be given, resulting in multiple BLEU scores.
    >>> weights = [
    ...     (0.5, 0.5),
    ...     (0.333, 0.333, 0.334),
    ...     (0.25, 0.25, 0.25, 0.25),
    ...     (0.2, 0.2, 0.2, 0.2, 0.2)
    ... ]
    >>> corpus_bleu(list_of_references, hypotheses, weights=weights) # doctest: +ELLIPSIS
    [0.8242..., 0.7067..., 0.5920..., 0.4719...]

    :param list_of_references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type list_of_references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on (one or a list of weights)
    :type weights: tuple(float) / list(tuple(float))
    :param smoothing_function:
    :type smoothing_function: SmoothingFunction
    :param auto_reweigh: Option to re-normalize the weights uniformly.
    :type auto_reweigh: bool
    :return: The corpus-level BLEU score.
    :rtype: float
    """
    p_numerators = Counter()
    p_denominators = Counter()
    hyp_lengths, ref_lengths = (0, 0)
    assert len(list_of_references) == len(hypotheses), 'The number of hypotheses and their reference(s) should be the same '
    try:
        weights[0][0]
    except TypeError:
        weights = [weights]
    max_weight_length = max((len(weight) for weight in weights))
    for references, hypothesis in zip(list_of_references, hypotheses):
        for i in range(1, max_weight_length + 1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)
    bp = brevity_penalty(ref_lengths, hyp_lengths)
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False) for i in range(1, max_weight_length + 1)]
    if p_numerators[1] == 0:
        return 0 if len(weights) == 1 else [0] * len(weights)
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths)
    bleu_scores = []
    for weight in weights:
        if auto_reweigh:
            if hyp_lengths < 4 and weight == (0.25, 0.25, 0.25, 0.25):
                weight = (1 / hyp_lengths,) * hyp_lengths
        s = (w_i * math.log(p_i) for w_i, p_i in zip(weight, p_n) if p_i > 0)
        s = bp * math.exp(math.fsum(s))
        bleu_scores.append(s)
    return bleu_scores[0] if len(weights) == 1 else bleu_scores