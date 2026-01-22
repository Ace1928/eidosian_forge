import os
import tempfile
from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.classify.megam import call_megam, parse_megam_weights, write_megam_file
from nltk.classify.tadm import call_tadm, parse_tadm_weights, write_tadm_file
from nltk.classify.util import CutoffChecker, accuracy, log_likelihood
from nltk.data import gzip_open_unicode
from nltk.probability import DictionaryProbDist
from nltk.util import OrderedDict
def calculate_deltas(train_toks, classifier, unattested, ffreq_empirical, nfmap, nfarray, nftranspose, encoding):
    """
    Calculate the update values for the classifier weights for
    this iteration of IIS.  These update weights are the value of
    ``delta`` that solves the equation::

      ffreq_empirical[i]
             =
      SUM[fs,l] (classifier.prob_classify(fs).prob(l) *
                 feature_vector(fs,l)[i] *
                 exp(delta[i] * nf(feature_vector(fs,l))))

    Where:
        - *(fs,l)* is a (featureset, label) tuple from ``train_toks``
        - *feature_vector(fs,l)* = ``encoding.encode(fs,l)``
        - *nf(vector)* = ``sum([val for (id,val) in vector])``

    This method uses Newton's method to solve this equation for
    *delta[i]*.  In particular, it starts with a guess of
    ``delta[i]`` = 1; and iteratively updates ``delta`` with:

    | delta[i] -= (ffreq_empirical[i] - sum1[i])/(-sum2[i])

    until convergence, where *sum1* and *sum2* are defined as:

    |    sum1[i](delta) = SUM[fs,l] f[i](fs,l,delta)
    |    sum2[i](delta) = SUM[fs,l] (f[i](fs,l,delta).nf(feature_vector(fs,l)))
    |    f[i](fs,l,delta) = (classifier.prob_classify(fs).prob(l) .
    |                        feature_vector(fs,l)[i] .
    |                        exp(delta[i] . nf(feature_vector(fs,l))))

    Note that *sum1* and *sum2* depend on ``delta``; so they need
    to be re-computed each iteration.

    The variables ``nfmap``, ``nfarray``, and ``nftranspose`` are
    used to generate a dense encoding for *nf(ltext)*.  This
    allows ``_deltas`` to calculate *sum1* and *sum2* using
    matrices, which yields a significant performance improvement.

    :param train_toks: The set of training tokens.
    :type train_toks: list(tuple(dict, str))
    :param classifier: The current classifier.
    :type classifier: ClassifierI
    :param ffreq_empirical: An array containing the empirical
        frequency for each feature.  The *i*\\ th element of this
        array is the empirical frequency for feature *i*.
    :type ffreq_empirical: sequence of float
    :param unattested: An array that is 1 for features that are
        not attested in the training data; and 0 for features that
        are attested.  In other words, ``unattested[i]==0`` iff
        ``ffreq_empirical[i]==0``.
    :type unattested: sequence of int
    :param nfmap: A map that can be used to compress ``nf`` to a dense
        vector.
    :type nfmap: dict(int -> int)
    :param nfarray: An array that can be used to uncompress ``nf``
        from a dense vector.
    :type nfarray: array(float)
    :param nftranspose: The transpose of ``nfarray``
    :type nftranspose: array(float)
    """
    NEWTON_CONVERGE = 1e-12
    MAX_NEWTON = 300
    deltas = numpy.ones(encoding.length(), 'd')
    A = numpy.zeros((len(nfmap), encoding.length()), 'd')
    for tok, label in train_toks:
        dist = classifier.prob_classify(tok)
        for label in encoding.labels():
            feature_vector = encoding.encode(tok, label)
            nf = sum((val for id, val in feature_vector))
            for id, val in feature_vector:
                A[nfmap[nf], id] += dist.prob(label) * val
    A /= len(train_toks)
    for rangenum in range(MAX_NEWTON):
        nf_delta = numpy.outer(nfarray, deltas)
        exp_nf_delta = 2 ** nf_delta
        nf_exp_nf_delta = nftranspose * exp_nf_delta
        sum1 = numpy.sum(exp_nf_delta * A, axis=0)
        sum2 = numpy.sum(nf_exp_nf_delta * A, axis=0)
        for fid in unattested:
            sum2[fid] += 1
        deltas -= (ffreq_empirical - sum1) / -sum2
        n_error = numpy.sum(abs(ffreq_empirical - sum1)) / numpy.sum(abs(deltas))
        if n_error < NEWTON_CONVERGE:
            return deltas
    return deltas