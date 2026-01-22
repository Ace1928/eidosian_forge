from collections.abc import Iterable, Sequence
import wandb
from wandb import util
from wandb.sdk.lib import deprecate
def encode_labels(df):
    pd = util.get_module('pandas', required='Logging dataframes requires pandas')
    preprocessing = util.get_module('sklearn.preprocessing', 'roc requires the scikit preprocessing submodule, install with `pip install scikit-learn`')
    le = preprocessing.LabelEncoder()
    categorical_cols = df.select_dtypes(exclude=['int', 'float', 'float64', 'float32', 'int32', 'int64']).columns
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))