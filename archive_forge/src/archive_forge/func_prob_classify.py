from nltk.internals import overridden
def prob_classify(self, featureset):
    """
        :return: a probability distribution over sets of labels for the
            given featureset.
        :rtype: ProbDistI
        """
    if overridden(self.prob_classify_many):
        return self.prob_classify_many([featureset])[0]
    else:
        raise NotImplementedError()