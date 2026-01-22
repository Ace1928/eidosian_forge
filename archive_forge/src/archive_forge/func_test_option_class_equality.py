from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
@pytest.mark.filterwarnings('ignore:pyarrow.CumulativeSumOptions is deprecated as of 14.0')
def test_option_class_equality():
    options = [pc.ArraySortOptions(), pc.AssumeTimezoneOptions('UTC'), pc.CastOptions.safe(pa.int8()), pc.CountOptions(), pc.DayOfWeekOptions(count_from_zero=False, week_start=0), pc.DictionaryEncodeOptions(), pc.RunEndEncodeOptions(), pc.ElementWiseAggregateOptions(skip_nulls=True), pc.ExtractRegexOptions('pattern'), pc.FilterOptions(), pc.IndexOptions(pa.scalar(1)), pc.JoinOptions(), pc.ListSliceOptions(0, -1, 1, True), pc.MakeStructOptions(['field', 'names'], field_nullability=[True, True], field_metadata=[pa.KeyValueMetadata({'a': '1'}), pa.KeyValueMetadata({'b': '2'})]), pc.MapLookupOptions(pa.scalar(1), 'first'), pc.MatchSubstringOptions('pattern'), pc.ModeOptions(), pc.NullOptions(), pc.PadOptions(5), pc.PairwiseOptions(period=1), pc.PartitionNthOptions(1, null_placement='at_start'), pc.CumulativeOptions(start=None, skip_nulls=False), pc.QuantileOptions(), pc.RandomOptions(), pc.RankOptions(sort_keys='ascending', null_placement='at_start', tiebreaker='max'), pc.ReplaceSliceOptions(0, 1, 'a'), pc.ReplaceSubstringOptions('a', 'b'), pc.RoundOptions(2, 'towards_infinity'), pc.RoundBinaryOptions('towards_infinity'), pc.RoundTemporalOptions(1, 'second', week_starts_monday=True), pc.RoundToMultipleOptions(100, 'towards_infinity'), pc.ScalarAggregateOptions(), pc.SelectKOptions(0, sort_keys=[('b', 'ascending')]), pc.SetLookupOptions(pa.array([1])), pc.SliceOptions(0, 1, 1), pc.SortOptions([('dummy', 'descending')], null_placement='at_start'), pc.SplitOptions(), pc.SplitPatternOptions('pattern'), pc.StrftimeOptions(), pc.StrptimeOptions('%Y', 's', True), pc.StructFieldOptions(indices=[]), pc.TakeOptions(), pc.TDigestOptions(), pc.TrimOptions(' '), pc.Utf8NormalizeOptions('NFKC'), pc.VarianceOptions(), pc.WeekOptions(week_starts_monday=True, count_from_zero=False, first_week_is_fully_in_year=False)]
    if sys.platform != 'win32' or util.windows_has_tzdata():
        options.append(pc.AssumeTimezoneOptions('Europe/Ljubljana'))
    classes = {type(option) for option in options}
    for cls in exported_option_classes:
        if cls not in classes and (sys.platform != 'win32' or util.windows_has_tzdata()) and (cls != pc.AssumeTimezoneOptions):
            try:
                options.append(cls())
            except TypeError:
                pytest.fail(f'Options class is not tested: {cls}')
    for option in options:
        assert option == option
        assert repr(option).startswith(option.__class__.__name__)
        buf = option.serialize()
        deserialized = pc.FunctionOptions.deserialize(buf)
        assert option == deserialized
        if repr(option).startswith('CumulativeSumOptions'):
            assert repr(deserialized).startswith('CumulativeOptions')
        else:
            assert repr(option) == repr(deserialized)
    for option1, option2 in zip(options, options[1:]):
        assert option1 != option2
    assert repr(pc.IndexOptions(pa.scalar(1))) == 'IndexOptions(value=int64:1)'
    assert repr(pc.ArraySortOptions()) == 'ArraySortOptions(order=Ascending, null_placement=AtEnd)'