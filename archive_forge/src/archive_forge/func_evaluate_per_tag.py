from abc import ABCMeta, abstractmethod
from functools import lru_cache
from itertools import chain
from typing import Dict
from nltk.internals import deprecated, overridden
from nltk.metrics import ConfusionMatrix, accuracy
from nltk.tag.util import untag
def evaluate_per_tag(self, gold, alpha=0.5, truncate=None, sort_by_count=False):
    """Tabulate the **recall**, **precision** and **f-measure**
        for each tag from ``gold`` or from running ``tag`` on the tokenized
        sentences from ``gold``.

        >>> from nltk.tag import PerceptronTagger
        >>> from nltk.corpus import treebank
        >>> tagger = PerceptronTagger()
        >>> gold_data = treebank.tagged_sents()[:10]
        >>> print(tagger.evaluate_per_tag(gold_data))
           Tag | Prec.  | Recall | F-measure
        -------+--------+--------+-----------
            '' | 1.0000 | 1.0000 | 1.0000
             , | 1.0000 | 1.0000 | 1.0000
        -NONE- | 0.0000 | 0.0000 | 0.0000
             . | 1.0000 | 1.0000 | 1.0000
            CC | 1.0000 | 1.0000 | 1.0000
            CD | 0.7143 | 1.0000 | 0.8333
            DT | 1.0000 | 1.0000 | 1.0000
            EX | 1.0000 | 1.0000 | 1.0000
            IN | 0.9167 | 0.8800 | 0.8980
            JJ | 0.8889 | 0.8889 | 0.8889
           JJR | 0.0000 | 0.0000 | 0.0000
           JJS | 1.0000 | 1.0000 | 1.0000
            MD | 1.0000 | 1.0000 | 1.0000
            NN | 0.8000 | 0.9333 | 0.8615
           NNP | 0.8929 | 1.0000 | 0.9434
           NNS | 0.9500 | 1.0000 | 0.9744
           POS | 1.0000 | 1.0000 | 1.0000
           PRP | 1.0000 | 1.0000 | 1.0000
          PRP$ | 1.0000 | 1.0000 | 1.0000
            RB | 0.4000 | 1.0000 | 0.5714
           RBR | 1.0000 | 0.5000 | 0.6667
            RP | 1.0000 | 1.0000 | 1.0000
            TO | 1.0000 | 1.0000 | 1.0000
            VB | 1.0000 | 1.0000 | 1.0000
           VBD | 0.8571 | 0.8571 | 0.8571
           VBG | 1.0000 | 0.8000 | 0.8889
           VBN | 1.0000 | 0.8000 | 0.8889
           VBP | 1.0000 | 1.0000 | 1.0000
           VBZ | 1.0000 | 1.0000 | 1.0000
           WDT | 0.0000 | 0.0000 | 0.0000
            `` | 1.0000 | 1.0000 | 1.0000
        <BLANKLINE>

        :param gold: The list of tagged sentences to score the tagger on.
        :type gold: list(list(tuple(str, str)))
        :param alpha: Ratio of the cost of false negative compared to false
            positives, as used in the f-measure computation. Defaults to 0.5,
            where the costs are equal.
        :type alpha: float
        :param truncate: If specified, then only show the specified
            number of values.  Any sorting (e.g., sort_by_count)
            will be performed before truncation. Defaults to None
        :type truncate: int, optional
        :param sort_by_count: Whether to sort the outputs on number of
            occurrences of that tag in the ``gold`` data, defaults to False
        :type sort_by_count: bool, optional
        :return: A tabulated recall, precision and f-measure string
        :rtype: str
        """
    cm = self.confusion(gold)
    return cm.evaluate(alpha=alpha, truncate=truncate, sort_by_count=sort_by_count)