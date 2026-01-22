import warnings
from collections import defaultdict
from math import log
@distortion_factor.setter
def distortion_factor(self, d):
    self.__distortion_factor = d
    self.__compute_log_distortion()