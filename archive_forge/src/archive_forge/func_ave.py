from .._utils import groupby_apply
from ..doctools import document
from ..mapping.aes import ALL_AESTHETICS
from ..mapping.evaluation import after_stat
from .stat import stat
def ave(df):
    """
            Calculate proportion values
            """
    df['prop'] = df['n'] / df['n'].sum()
    return df