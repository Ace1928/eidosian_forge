import math
from itertools import islice
from nltk.util import choose, ngrams
def position_of_ngram(ngram, sentence):
    """
    This function returns the position of the first instance of the ngram
    appearing in a sentence.

    Note that one could also use string as follows but the code is a little
    convoluted with type casting back and forth:

        char_pos = ' '.join(sent)[:' '.join(sent).index(' '.join(ngram))]
        word_pos = char_pos.count(' ')

    Another way to conceive this is:

        return next(i for i, ng in enumerate(ngrams(sentence, len(ngram)))
                    if ng == ngram)

    :param ngram: The ngram that needs to be searched
    :type ngram: tuple
    :param sentence: The list of tokens to search from.
    :type sentence: list(str)
    """
    for i, sublist in enumerate(ngrams(sentence, len(ngram))):
        if ngram == sublist:
            return i