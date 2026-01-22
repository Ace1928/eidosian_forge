from math import sqrt
def get_words_from_dictionary(lemmas):
    """
    Get original set of words used for analysis.

    :param lemmas: A dictionary where keys are lemmas and values are sets
        or lists of words corresponding to that lemma.
    :type lemmas: dict(str): list(str)
    :return: Set of words that exist as values in the dictionary
    :rtype: set(str)
    """
    words = set()
    for lemma in lemmas:
        words.update(set(lemmas[lemma]))
    return words