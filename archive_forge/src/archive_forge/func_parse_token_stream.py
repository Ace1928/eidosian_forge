import math
def parse_token_stream(stream, soft_delimiter, hard_delimiter):
    """Parses a stream of tokens and splits it into sentences (using C{soft_delimiter} tokens)
    and blocks (using C{hard_delimiter} tokens) for use with the L{align_texts} function.
    """
    return [[sum((len(token) for token in sentence_it)) for sentence_it in split_at(block_it, soft_delimiter)] for block_it in split_at(stream, hard_delimiter)]