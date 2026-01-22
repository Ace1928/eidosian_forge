import fractions
import math
from collections import Counter
from nltk.util import ngrams
def corpus_nist(list_of_references, hypotheses, n=5):
    """
    Calculate a single corpus-level NIST score (aka. system-level BLEU) for all
    the hypotheses and their respective references.

    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param n: highest n-gram order
    :type n: int
    """
    assert len(list_of_references) == len(hypotheses), 'The number of hypotheses and their reference(s) should be the same'
    ngram_freq = Counter()
    total_reference_words = 0
    for references in list_of_references:
        for reference in references:
            for i in range(1, n + 1):
                ngram_freq.update(ngrams(reference, i))
            total_reference_words += len(reference)
    information_weights = {}
    for _ngram in ngram_freq:
        _mgram = _ngram[:-1]
        if _mgram and _mgram in ngram_freq:
            numerator = ngram_freq[_mgram]
        else:
            numerator = total_reference_words
        information_weights[_ngram] = math.log(numerator / ngram_freq[_ngram], 2)
    nist_precision_numerator_per_ngram = Counter()
    nist_precision_denominator_per_ngram = Counter()
    l_ref, l_sys = (0, 0)
    for i in range(1, n + 1):
        for references, hypothesis in zip(list_of_references, hypotheses):
            hyp_len = len(hypothesis)
            nist_score_per_ref = []
            for reference in references:
                _ref_len = len(reference)
                hyp_ngrams = Counter(ngrams(hypothesis, i)) if len(hypothesis) >= i else Counter()
                ref_ngrams = Counter(ngrams(reference, i)) if len(reference) >= i else Counter()
                ngram_overlaps = hyp_ngrams & ref_ngrams
                _numerator = sum((information_weights[_ngram] * count for _ngram, count in ngram_overlaps.items()))
                _denominator = sum(hyp_ngrams.values())
                _precision = 0 if _denominator == 0 else _numerator / _denominator
                nist_score_per_ref.append((_precision, _numerator, _denominator, _ref_len))
            precision, numerator, denominator, ref_len = max(nist_score_per_ref)
            nist_precision_numerator_per_ngram[i] += numerator
            nist_precision_denominator_per_ngram[i] += denominator
            l_ref += ref_len
            l_sys += hyp_len
    nist_precision = 0
    for i in nist_precision_numerator_per_ngram:
        precision = nist_precision_numerator_per_ngram[i] / nist_precision_denominator_per_ngram[i]
        nist_precision += precision
    return nist_precision * nist_length_penalty(l_ref, l_sys)