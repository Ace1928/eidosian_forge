from abc import ABCMeta, abstractmethod
from functools import lru_cache
from itertools import chain
from typing import Dict
from nltk.internals import deprecated, overridden
from nltk.metrics import ConfusionMatrix, accuracy
from nltk.tag.util import untag
def confusion(self, gold):
    """
        Return a ConfusionMatrix with the tags from ``gold`` as the reference
        values, with the predictions from ``tag_sents`` as the predicted values.

        >>> from nltk.tag import PerceptronTagger
        >>> from nltk.corpus import treebank
        >>> tagger = PerceptronTagger()
        >>> gold_data = treebank.tagged_sents()[:10]
        >>> print(tagger.confusion(gold_data))
               |        -                                                                                     |
               |        N                                                                                     |
               |        O                                               P                                     |
               |        N                       J  J        N  N  P  P  R     R           V  V  V  V  V  W    |
               |  '     E     C  C  D  E  I  J  J  J  M  N  N  N  O  R  P  R  B  R  T  V  B  B  B  B  B  D  ` |
               |  '  ,  -  .  C  D  T  X  N  J  R  S  D  N  P  S  S  P  $  B  R  P  O  B  D  G  N  P  Z  T  ` |
        -------+----------------------------------------------------------------------------------------------+
            '' | <1> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
             , |  .<15> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
        -NONE- |  .  . <.> .  .  2  .  .  .  2  .  .  .  5  1  .  .  .  .  2  .  .  .  .  .  .  .  .  .  .  . |
             . |  .  .  .<10> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
            CC |  .  .  .  . <1> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
            CD |  .  .  .  .  . <5> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
            DT |  .  .  .  .  .  .<20> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
            EX |  .  .  .  .  .  .  . <1> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
            IN |  .  .  .  .  .  .  .  .<22> .  .  .  .  .  .  .  .  .  .  3  .  .  .  .  .  .  .  .  .  .  . |
            JJ |  .  .  .  .  .  .  .  .  .<16> .  .  .  .  1  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  . |
           JJR |  .  .  .  .  .  .  .  .  .  . <.> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
           JJS |  .  .  .  .  .  .  .  .  .  .  . <1> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
            MD |  .  .  .  .  .  .  .  .  .  .  .  . <1> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
            NN |  .  .  .  .  .  .  .  .  .  .  .  .  .<28> 1  1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
           NNP |  .  .  .  .  .  .  .  .  .  .  .  .  .  .<25> .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
           NNS |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .<19> .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
           POS |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <1> .  .  .  .  .  .  .  .  .  .  .  .  .  . |
           PRP |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <4> .  .  .  .  .  .  .  .  .  .  .  .  . |
          PRP$ |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <2> .  .  .  .  .  .  .  .  .  .  .  . |
            RB |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <4> .  .  .  .  .  .  .  .  .  .  . |
           RBR |  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  . <1> .  .  .  .  .  .  .  .  .  . |
            RP |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <1> .  .  .  .  .  .  .  .  . |
            TO |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <5> .  .  .  .  .  .  .  . |
            VB |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <3> .  .  .  .  .  .  . |
           VBD |  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  . <6> .  .  .  .  .  . |
           VBG |  .  .  .  .  .  .  .  .  .  .  .  .  .  1  .  .  .  .  .  .  .  .  .  .  . <4> .  .  .  .  . |
           VBN |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  1  . <4> .  .  .  . |
           VBP |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <3> .  .  . |
           VBZ |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <7> .  . |
           WDT |  .  .  .  .  .  .  .  .  2  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <.> . |
            `` |  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . <1>|
        -------+----------------------------------------------------------------------------------------------+
        (row = reference; col = test)
        <BLANKLINE>

        :param gold: The list of tagged sentences to run the tagger with,
            also used as the reference values in the generated confusion matrix.
        :type gold: list(list(tuple(str, str)))
        :rtype: ConfusionMatrix
        """
    return self._confusion_cached(tuple((tuple(sent) for sent in gold)))