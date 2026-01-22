from functools import partial
from itertools import chain
from nltk.util import everygrams, pad_sequence
def padded_everygram_pipeline(order, text):
    """Default preprocessing for a sequence of sentences.

    Creates two iterators:

    - sentences padded and turned into sequences of `nltk.util.everygrams`
    - sentences padded as above and chained together for a flat stream of words

    :param order: Largest ngram length produced by `everygrams`.
    :param text: Text to iterate over. Expected to be an iterable of sentences.
    :type text: Iterable[Iterable[str]]
    :return: iterator over text as ngrams, iterator over text as vocabulary data
    """
    padding_fn = partial(pad_both_ends, n=order)
    return ((everygrams(list(padding_fn(sent)), max_len=order) for sent in text), flatten(map(padding_fn, text)))