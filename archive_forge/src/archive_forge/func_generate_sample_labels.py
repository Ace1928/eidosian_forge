import numpy as np
import scipy
import sklearn
import meld
import graphtools as gt
def generate_sample_labels(self):
    np.random.seed(self.seed)
    self.sample_indicator = np.random.binomial(1, self.pdf)
    self.sample_labels = np.array(['ctrl' if ind == 0 else 'expt' for ind in self.sample_indicator])