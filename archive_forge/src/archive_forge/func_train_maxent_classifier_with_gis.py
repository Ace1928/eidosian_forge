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
def train_maxent_classifier_with_gis(train_toks, trace=3, encoding=None, labels=None, **cutoffs):
    """
    Train a new ``ConditionalExponentialClassifier``, using the given
    training samples, using the Generalized Iterative Scaling
    algorithm.  This ``ConditionalExponentialClassifier`` will encode
    the model that maximizes entropy from all the models that are
    empirically consistent with ``train_toks``.

    :see: ``train_maxent_classifier()`` for parameter descriptions.
    """
    cutoffs.setdefault('max_iter', 100)
    cutoffchecker = CutoffChecker(cutoffs)
    if encoding is None:
        encoding = GISEncoding.train(train_toks, labels=labels)
    if not hasattr(encoding, 'C'):
        raise TypeError('The GIS algorithm requires an encoding that defines C (e.g., GISEncoding).')
    Cinv = 1.0 / encoding.C
    empirical_fcount = calculate_empirical_fcount(train_toks, encoding)
    unattested = set(numpy.nonzero(empirical_fcount == 0)[0])
    weights = numpy.zeros(len(empirical_fcount), 'd')
    for fid in unattested:
        weights[fid] = numpy.NINF
    classifier = ConditionalExponentialClassifier(encoding, weights)
    log_empirical_fcount = numpy.log2(empirical_fcount)
    del empirical_fcount
    if trace > 0:
        print('  ==> Training (%d iterations)' % cutoffs['max_iter'])
    if trace > 2:
        print()
        print('      Iteration    Log Likelihood    Accuracy')
        print('      ---------------------------------------')
    try:
        while True:
            if trace > 2:
                ll = cutoffchecker.ll or log_likelihood(classifier, train_toks)
                acc = cutoffchecker.acc or accuracy(classifier, train_toks)
                iternum = cutoffchecker.iter
                print('     %9d    %14.5f    %9.3f' % (iternum, ll, acc))
            estimated_fcount = calculate_estimated_fcount(classifier, train_toks, encoding)
            for fid in unattested:
                estimated_fcount[fid] += 1
            log_estimated_fcount = numpy.log2(estimated_fcount)
            del estimated_fcount
            weights = classifier.weights()
            weights += (log_empirical_fcount - log_estimated_fcount) * Cinv
            classifier.set_weights(weights)
            if cutoffchecker.check(classifier, train_toks):
                break
    except KeyboardInterrupt:
        print('      Training stopped: keyboard interrupt')
    except:
        raise
    if trace > 2:
        ll = log_likelihood(classifier, train_toks)
        acc = accuracy(classifier, train_toks)
        print(f'         Final    {ll:14.5f}    {acc:9.3f}')
    return classifier