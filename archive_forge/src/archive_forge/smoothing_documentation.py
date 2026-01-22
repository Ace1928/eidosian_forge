from operator import methodcaller
from nltk.lm.api import Smoothing
from nltk.probability import ConditionalFreqDist
Count continuations that end with context and word.

        Continuations track unique ngram "types", regardless of how many
        instances were observed for each "type".
        This is different than raw ngram counts which track number of instances.
        