from numpy.testing import assert_equal
import numpy as np
def ttest_conditional_effect(self, factorind):
    if factorind == 1:
        return (self.resols.t_test(self.C1), self.C1_label)
    else:
        return (self.resols.t_test(self.C2), self.C2_label)