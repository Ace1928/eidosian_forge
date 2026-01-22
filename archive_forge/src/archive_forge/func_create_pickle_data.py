from datetime import timedelta
import os
import pickle
import platform as pl
import sys
import numpy as np
import pandas
from pandas import (
from pandas.arrays import SparseArray
from pandas.tseries.offsets import (
def create_pickle_data():
    """create the pickle data"""
    data = {'A': [0.0, 1.0, 2.0, 3.0, np.nan], 'B': [0, 1, 0, 1, 0], 'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'D': date_range('1/1/2009', periods=5), 'E': [0.0, 1, Timestamp('20100101'), 'foo', 2.0]}
    scalars = {'timestamp': Timestamp('20130101'), 'period': Period('2012', 'M')}
    index = {'int': Index(np.arange(10)), 'date': date_range('20130101', periods=10), 'period': period_range('2013-01-01', freq='M', periods=10), 'float': Index(np.arange(10, dtype=np.float64)), 'uint': Index(np.arange(10, dtype=np.uint64)), 'timedelta': timedelta_range('00:00:00', freq='30min', periods=10)}
    index['range'] = RangeIndex(10)
    index['interval'] = interval_range(0, periods=10)
    mi = {'reg2': MultiIndex.from_tuples(tuple(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])), names=['first', 'second'])}
    series = {'float': Series(data['A']), 'int': Series(data['B']), 'mixed': Series(data['E']), 'ts': Series(np.arange(10).astype(np.int64), index=date_range('20130101', periods=10)), 'mi': Series(np.arange(5).astype(np.float64), index=MultiIndex.from_tuples(tuple(zip(*[[1, 1, 2, 2, 2], [3, 4, 3, 4, 5]])), names=['one', 'two'])), 'dup': Series(np.arange(5).astype(np.float64), index=['A', 'B', 'C', 'D', 'A']), 'cat': Series(Categorical(['foo', 'bar', 'baz'])), 'dt': Series(date_range('20130101', periods=5)), 'dt_tz': Series(date_range('20130101', periods=5, tz='US/Eastern')), 'period': Series([Period('2000Q1')] * 5)}
    mixed_dup_df = DataFrame(data)
    mixed_dup_df.columns = list('ABCDA')
    frame = {'float': DataFrame({'A': series['float'], 'B': series['float'] + 1}), 'int': DataFrame({'A': series['int'], 'B': series['int'] + 1}), 'mixed': DataFrame({k: data[k] for k in ['A', 'B', 'C', 'D']}), 'mi': DataFrame({'A': np.arange(5).astype(np.float64), 'B': np.arange(5).astype(np.int64)}, index=MultiIndex.from_tuples(tuple(zip(*[['bar', 'bar', 'baz', 'baz', 'baz'], ['one', 'two', 'one', 'two', 'three']])), names=['first', 'second'])), 'dup': DataFrame(np.arange(15).reshape(5, 3).astype(np.float64), columns=['A', 'B', 'A']), 'cat_onecol': DataFrame({'A': Categorical(['foo', 'bar'])}), 'cat_and_float': DataFrame({'A': Categorical(['foo', 'bar', 'baz']), 'B': np.arange(3).astype(np.int64)}), 'mixed_dup': mixed_dup_df, 'dt_mixed_tzs': DataFrame({'A': Timestamp('20130102', tz='US/Eastern'), 'B': Timestamp('20130603', tz='CET')}, index=range(5)), 'dt_mixed2_tzs': DataFrame({'A': Timestamp('20130102', tz='US/Eastern'), 'B': Timestamp('20130603', tz='CET'), 'C': Timestamp('20130603', tz='UTC')}, index=range(5))}
    cat = {'int8': Categorical(list('abcdefg')), 'int16': Categorical(np.arange(1000)), 'int32': Categorical(np.arange(10000))}
    timestamp = {'normal': Timestamp('2011-01-01'), 'nat': NaT, 'tz': Timestamp('2011-01-01', tz='US/Eastern')}
    off = {'DateOffset': DateOffset(years=1), 'DateOffset_h_ns': DateOffset(hour=6, nanoseconds=5824), 'BusinessDay': BusinessDay(offset=timedelta(seconds=9)), 'BusinessHour': BusinessHour(normalize=True, n=6, end='15:14'), 'CustomBusinessDay': CustomBusinessDay(weekmask='Mon Fri'), 'SemiMonthBegin': SemiMonthBegin(day_of_month=9), 'SemiMonthEnd': SemiMonthEnd(day_of_month=24), 'MonthBegin': MonthBegin(1), 'MonthEnd': MonthEnd(1), 'QuarterBegin': QuarterBegin(1), 'QuarterEnd': QuarterEnd(1), 'Day': Day(1), 'YearBegin': YearBegin(1), 'YearEnd': YearEnd(1), 'Week': Week(1), 'Week_Tues': Week(2, normalize=False, weekday=1), 'WeekOfMonth': WeekOfMonth(week=3, weekday=4), 'LastWeekOfMonth': LastWeekOfMonth(n=1, weekday=3), 'FY5253': FY5253(n=2, weekday=6, startingMonth=7, variation='last'), 'Easter': Easter(), 'Hour': Hour(1), 'Minute': Minute(1)}
    return {'series': series, 'frame': frame, 'index': index, 'scalars': scalars, 'mi': mi, 'sp_series': {'float': _create_sp_series(), 'ts': _create_sp_tsseries()}, 'sp_frame': {'float': _create_sp_frame()}, 'cat': cat, 'timestamp': timestamp, 'offsets': off}