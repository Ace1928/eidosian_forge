from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def batch_tag_incremental(self, sequences, gold):
    """
        Tags by applying each rule to the entire corpus (rather than all rules to a
        single sequence). The point is to collect statistics on the test set for
        individual rules.

        NOTE: This is inefficient (does not build any index, so will traverse the entire
        corpus N times for N rules) -- usually you would not care about statistics for
        individual rules and thus use batch_tag() instead

        :param sequences: lists of token sequences (sentences, in some applications) to be tagged
        :type sequences: list of list of strings
        :param gold: the gold standard
        :type gold: list of list of strings
        :returns: tuple of (tagged_sequences, ordered list of rule scores (one for each rule))
        """

    def counterrors(xs):
        return sum((t[1] != g[1] for pair in zip(xs, gold) for t, g in zip(*pair)))
    testing_stats = {}
    testing_stats['tokencount'] = sum((len(t) for t in sequences))
    testing_stats['sequencecount'] = len(sequences)
    tagged_tokenses = [self._initial_tagger.tag(tokens) for tokens in sequences]
    testing_stats['initialerrors'] = counterrors(tagged_tokenses)
    testing_stats['initialacc'] = 1 - testing_stats['initialerrors'] / testing_stats['tokencount']
    errors = [testing_stats['initialerrors']]
    for rule in self._rules:
        for tagged_tokens in tagged_tokenses:
            rule.apply(tagged_tokens)
        errors.append(counterrors(tagged_tokenses))
    testing_stats['rulescores'] = [err0 - err1 for err0, err1 in zip(errors, errors[1:])]
    testing_stats['finalerrors'] = errors[-1]
    testing_stats['finalacc'] = 1 - testing_stats['finalerrors'] / testing_stats['tokencount']
    return (tagged_tokenses, testing_stats)