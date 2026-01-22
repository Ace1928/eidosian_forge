import copy
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
from ._typing import BoosterParam, Callable, FPreProcCallable
from .callback import (
from .compat import SKLEARN_INSTALLED, DataFrame, XGBStratifiedKFold
from .core import (
def mknfold(dall: DMatrix, nfold: int, param: BoosterParam, seed: int, evals: Sequence[str]=(), fpreproc: Optional[FPreProcCallable]=None, stratified: Optional[bool]=False, folds: Optional[XGBStratifiedKFold]=None, shuffle: bool=True) -> List[CVPack]:
    """
    Make an n-fold list of CVPack from random indices.
    """
    evals = list(evals)
    np.random.seed(seed)
    if stratified is False and folds is None:
        if len(dall.get_uint_info('group_ptr')) > 1:
            return mkgroupfold(dall, nfold, param, evals=evals, fpreproc=fpreproc, shuffle=shuffle)
        if shuffle is True:
            idx = np.random.permutation(dall.num_row())
        else:
            idx = np.arange(dall.num_row())
        out_idset = np.array_split(idx, nfold)
        in_idset = [np.concatenate([out_idset[i] for i in range(nfold) if k != i]) for k in range(nfold)]
    elif folds is not None:
        try:
            in_idset = [x[0] for x in folds]
            out_idset = [x[1] for x in folds]
        except TypeError:
            splits = list(folds.split(X=dall.get_label(), y=dall.get_label()))
            in_idset = [x[0] for x in splits]
            out_idset = [x[1] for x in splits]
        nfold = len(out_idset)
    else:
        sfk = XGBStratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
        splits = list(sfk.split(X=dall.get_label(), y=dall.get_label()))
        in_idset = [x[0] for x in splits]
        out_idset = [x[1] for x in splits]
        nfold = len(out_idset)
    ret = []
    for k in range(nfold):
        dtrain = dall.slice(in_idset[k])
        dtest = dall.slice(out_idset[k])
        if fpreproc is not None:
            dtrain, dtest, tparam = fpreproc(dtrain, dtest, param.copy())
        else:
            tparam = param
        plst = list(tparam.items()) + [('eval_metric', itm) for itm in evals]
        ret.append(CVPack(dtrain, dtest, plst))
    return ret