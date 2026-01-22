import datetime
import pandas
import pytest
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_object_dtype
import modin.pandas as pd
from modin.tests.pandas.utils import df_equals
from modin.tests.pandas.utils import eval_io as general_eval_io
from modin.utils import try_cast_to_pandas
def run_and_compare(fn, data, data2=None, force_lazy=True, force_hdk_execute=False, force_arrow_execute=False, allow_subqueries=False, comparator=df_equals, **kwargs):
    """Verify equality of the results of the passed function executed against pandas and modin frame."""

    def run_modin(fn, data, data2, force_lazy, force_hdk_execute, force_arrow_execute, allow_subqueries, constructor_kwargs, **kwargs):
        kwargs['df1'] = pd.DataFrame(data, **constructor_kwargs)
        kwargs['df2'] = pd.DataFrame(data2, **constructor_kwargs)
        kwargs['df'] = kwargs['df1']
        if force_hdk_execute:
            set_execution_mode(kwargs['df1'], 'hdk')
            set_execution_mode(kwargs['df2'], 'hdk')
        elif force_arrow_execute:
            set_execution_mode(kwargs['df1'], 'arrow')
            set_execution_mode(kwargs['df2'], 'arrow')
        elif force_lazy:
            set_execution_mode(kwargs['df1'], 'lazy')
            set_execution_mode(kwargs['df2'], 'lazy')
        exp_res = fn(lib=pd, **kwargs)
        if force_hdk_execute:
            set_execution_mode(exp_res, 'hdk', allow_subqueries)
        elif force_arrow_execute:
            set_execution_mode(exp_res, 'arrow', allow_subqueries)
        elif force_lazy:
            set_execution_mode(exp_res, None, allow_subqueries)
        return exp_res
    constructor_kwargs = kwargs.pop('constructor_kwargs', {})
    try:
        kwargs['df1'] = pandas.DataFrame(data, **constructor_kwargs)
        kwargs['df2'] = pandas.DataFrame(data2, **constructor_kwargs)
        kwargs['df'] = kwargs['df1']
        ref_res = fn(lib=pandas, **kwargs)
    except Exception as err:
        with pytest.raises(type(err)):
            exp_res = run_modin(fn=fn, data=data, data2=data2, force_lazy=force_lazy, force_hdk_execute=force_hdk_execute, force_arrow_execute=force_arrow_execute, allow_subqueries=allow_subqueries, constructor_kwargs=constructor_kwargs, **kwargs)
            _ = exp_res.index
    else:
        exp_res = run_modin(fn=fn, data=data, data2=data2, force_lazy=force_lazy, force_hdk_execute=force_hdk_execute, force_arrow_execute=force_arrow_execute, allow_subqueries=allow_subqueries, constructor_kwargs=constructor_kwargs, **kwargs)
        comparator(ref_res, exp_res)