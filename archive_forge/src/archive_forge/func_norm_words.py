import io
import math
import os
import typing
import weakref
def norm_words(width, words):
    """Cut any word in pieces no longer than 'width'."""
    nwords = []
    word_lengths = []
    for w in words:
        wl_lst = char_lengths(w)
        wl = sum(wl_lst)
        if wl <= width:
            nwords.append(w)
            word_lengths.append(wl)
            continue
        n = len(wl_lst)
        while n > 0:
            wl = sum(wl_lst[:n])
            if wl <= width:
                nwords.append(w[:n])
                word_lengths.append(wl)
                w = w[n:]
                wl_lst = wl_lst[n:]
                n = len(wl_lst)
            else:
                n -= 1
    return (nwords, word_lengths)