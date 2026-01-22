from collections.abc import Iterable, Sequence
import wandb
from wandb import util
from wandb.sdk.lib import deprecate
def test_types(**kwargs):
    np = util.get_module('numpy', required='Logging plots requires numpy')
    pd = util.get_module('pandas', required='Logging dataframes requires pandas')
    scipy = util.get_module('scipy', required='Logging scipy matrices requires scipy')
    base = util.get_module('sklearn.base', 'roc requires the scikit base submodule, install with `pip install scikit-learn`')
    test_passed = True
    for k, v in kwargs.items():
        if k == 'X' or k == 'X_test' or k == 'y' or (k == 'y_test') or (k == 'y_true') or (k == 'y_probas') or (k == 'x_labels') or (k == 'y_labels') or (k == 'matrix_values'):
            if not isinstance(v, (Sequence, Iterable, np.ndarray, np.generic, pd.DataFrame, pd.Series, list)):
                wandb.termerror('%s is not an array. Please try again.' % k)
                test_passed = False
        if k == 'model':
            if not base.is_classifier(v) and (not base.is_regressor(v)):
                wandb.termerror('%s is not a classifier or regressor. Please try again.' % k)
                test_passed = False
        elif k == 'clf' or k == 'binary_clf':
            if not base.is_classifier(v):
                wandb.termerror('%s is not a classifier. Please try again.' % k)
                test_passed = False
        elif k == 'regressor':
            if not base.is_regressor(v):
                wandb.termerror('%s is not a regressor. Please try again.' % k)
                test_passed = False
        elif k == 'clusterer':
            if not getattr(v, '_estimator_type', None) == 'clusterer':
                wandb.termerror('%s is not a clusterer. Please try again.' % k)
                test_passed = False
    return test_passed