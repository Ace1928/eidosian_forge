import pytest
from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus
def test_rte_classification_with_megam(self):
    try:
        config_megam()
    except (LookupError, AttributeError) as e:
        pytest.skip('Skipping tests with dependencies on MEGAM')
    clf = rte_classifier('megam', sample_N=100)