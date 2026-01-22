from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import RegexpTokenizer
def rte_classifier(algorithm, sample_N=None):
    from nltk.corpus import rte as rte_corpus
    train_set = rte_corpus.pairs(['rte1_dev.xml', 'rte2_dev.xml', 'rte3_dev.xml'])
    test_set = rte_corpus.pairs(['rte1_test.xml', 'rte2_test.xml', 'rte3_test.xml'])
    if sample_N is not None:
        train_set = train_set[:sample_N]
        test_set = test_set[:sample_N]
    featurized_train_set = rte_featurize(train_set)
    featurized_test_set = rte_featurize(test_set)
    print('Training classifier...')
    if algorithm in ['megam']:
        clf = MaxentClassifier.train(featurized_train_set, algorithm)
    elif algorithm in ['GIS', 'IIS']:
        clf = MaxentClassifier.train(featurized_train_set, algorithm)
    else:
        err_msg = str("RTEClassifier only supports these algorithms:\n 'megam', 'GIS', 'IIS'.\n")
        raise Exception(err_msg)
    print('Testing classifier...')
    acc = accuracy(clf, featurized_test_set)
    print('Accuracy: %6.4f' % acc)
    return clf