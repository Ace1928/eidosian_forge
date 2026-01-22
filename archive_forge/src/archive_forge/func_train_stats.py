from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def train_stats(self, statistic=None):
    """
        Return a named statistic collected during training, or a dictionary of all
        available statistics if no name given

        :param statistic: name of statistic
        :type statistic: str
        :return: some statistic collected during training of this tagger
        :rtype: any (but usually a number)
        """
    if statistic is None:
        return self._training_stats
    else:
        return self._training_stats.get(statistic)