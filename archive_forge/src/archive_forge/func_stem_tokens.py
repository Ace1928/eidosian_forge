import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def stem_tokens(tokens):
    """
        Apply WordNetDB rules or Stem each token of tokens

        Args:
          tokens: List of tokens to apply WordNetDB rules or to stem

        Returns:
          List of final stems
        """
    for i, token in enumerate(tokens):
        if len(token) > 0:
            if len(token) > 3:
                if token in Rouge.WORDNET_KEY_VALUE:
                    token = Rouge.WORDNET_KEY_VALUE[token]
                else:
                    token = Rouge.STEMMER.stem(token)
                tokens[i] = token
    return tokens