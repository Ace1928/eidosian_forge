import warnings
from collections import defaultdict
from math import log
def threshold_prune(self):
    if not self.items:
        return
    threshold = self.items[0].score() + self.__log_beam_threshold
    for hypothesis in reversed(self.items):
        if hypothesis.score() < threshold:
            self.items.pop()
        else:
            break