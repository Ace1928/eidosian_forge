import collections
def is_eval(mode):
    return mode in [KerasModeKeys.TEST, EstimatorModeKeys.EVAL]