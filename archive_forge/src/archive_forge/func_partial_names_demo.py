import math
import nltk.classify.util  # for accuracy & log_likelihood
from nltk.util import LazyMap
def partial_names_demo(trainer, features=names_demo_features):
    import random
    from nltk.corpus import names
    male_names = names.words('male.txt')
    female_names = names.words('female.txt')
    random.seed(654321)
    random.shuffle(male_names)
    random.shuffle(female_names)
    positive = map(features, male_names[:2000])
    unlabeled = map(features, male_names[2000:2500] + female_names[:500])
    test = [(name, True) for name in male_names[2500:2750]] + [(name, False) for name in female_names[500:750]]
    random.shuffle(test)
    print('Training classifier...')
    classifier = trainer(positive, unlabeled)
    print('Testing classifier...')
    acc = accuracy(classifier, [(features(n), m) for n, m in test])
    print('Accuracy: %6.4f' % acc)
    try:
        test_featuresets = [features(n) for n, m in test]
        pdists = classifier.prob_classify_many(test_featuresets)
        ll = [pdist.logprob(gold) for (name, gold), pdist in zip(test, pdists)]
        print('Avg. log likelihood: %6.4f' % (sum(ll) / len(test)))
        print()
        print('Unseen Names      P(Male)  P(Female)\n' + '-' * 40)
        for (name, is_male), pdist in zip(test, pdists)[:5]:
            if is_male == True:
                fmt = '  %-15s *%6.4f   %6.4f'
            else:
                fmt = '  %-15s  %6.4f  *%6.4f'
            print(fmt % (name, pdist.prob(True), pdist.prob(False)))
    except NotImplementedError:
        pass
    return classifier