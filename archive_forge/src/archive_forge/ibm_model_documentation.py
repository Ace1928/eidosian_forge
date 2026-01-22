from bisect import insort_left
from collections import defaultdict
from copy import deepcopy
from math import ceil

        :return: Zero-indexed alignment, suitable for use in external
            ``nltk.translate`` modules like ``nltk.translate.Alignment``
        :rtype: list(tuple)
        