import collections
def is_predict(mode):
    return mode in [KerasModeKeys.PREDICT, EstimatorModeKeys.PREDICT]