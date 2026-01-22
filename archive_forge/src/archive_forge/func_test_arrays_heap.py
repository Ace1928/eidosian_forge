from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_arrays_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, 'heap_null_array', 'arrow::NullArray of length 2, offset 0, null count 2')
    check_heap_repr(gdb_arrow, 'heap_int32_array', 'arrow::Int32Array of length 4, offset 0, null count 1 = {[0] = -5, [1] = 6, [2] = null, [3] = 42}')
    check_heap_repr(gdb_arrow, 'heap_int32_array_no_nulls', 'arrow::Int32Array of length 4, offset 0, null count 0 = {[0] = -5, [1] = 6, [2] = 3, [3] = 42}')
    check_heap_repr(gdb_arrow, 'heap_int32_array_sliced_1_9', 'arrow::Int32Array of length 9, offset 1, unknown null count = {[0] = 2, [1] = -3, [2] = 4, [3] = null, [4] = -5, [5] = 6, [6] = -7, [7] = 8, [8] = null}')
    check_heap_repr(gdb_arrow, 'heap_int32_array_sliced_2_6', 'arrow::Int32Array of length 6, offset 2, unknown null count = {[0] = -3, [1] = 4, [2] = null, [3] = -5, [4] = 6, [5] = -7}')
    check_heap_repr(gdb_arrow, 'heap_int32_array_sliced_8_4', 'arrow::Int32Array of length 4, offset 8, unknown null count = {[0] = 8, [1] = null, [2] = -9, [3] = -10}')
    check_heap_repr(gdb_arrow, 'heap_int32_array_sliced_empty', 'arrow::Int32Array of length 0, offset 6, unknown null count')
    check_heap_repr(gdb_arrow, 'heap_double_array', 'arrow::DoubleArray of length 2, offset 0, null count 1 = {[0] = -1.5, [1] = null}')
    check_heap_repr(gdb_arrow, 'heap_float16_array', 'arrow::HalfFloatArray of length 2, offset 0, null count 0 = {[0] = 0.0, [1] = -1.5}')
    check_heap_repr(gdb_arrow, 'heap_bool_array', 'arrow::BooleanArray of length 18, offset 0, null count 6 = {[0] = false, [1] = false, [2] = true, [3] = true, [4] = null, [5] = null, [6] = false, [7] = false, [8] = true, [9] = true, [10] = null, [11] = null, [12] = false, [13] = false, [14] = true, [15] = true, [16] = null, [17] = null}')
    check_heap_repr(gdb_arrow, 'heap_bool_array_sliced_1_9', 'arrow::BooleanArray of length 9, offset 1, unknown null count = {[0] = false, [1] = true, [2] = true, [3] = null, [4] = null, [5] = false, [6] = false, [7] = true, [8] = true}')
    check_heap_repr(gdb_arrow, 'heap_bool_array_sliced_2_6', 'arrow::BooleanArray of length 6, offset 2, unknown null count = {[0] = true, [1] = true, [2] = null, [3] = null, [4] = false, [5] = false}')
    check_heap_repr(gdb_arrow, 'heap_bool_array_sliced_empty', 'arrow::BooleanArray of length 0, offset 6, unknown null count')
    check_heap_repr(gdb_arrow, 'heap_date32_array', 'arrow::Date32Array of length 6, offset 0, null count 1 = {[0] = 0d [1970-01-01], [1] = null, [2] = 18336d [2020-03-15], [3] = -9004d [1945-05-08], [4] = -719162d [0001-01-01], [5] = -719163d [year <= 0]}')
    check_heap_repr(gdb_arrow, 'heap_date64_array', 'arrow::Date64Array of length 5, offset 0, null count 0 = {[0] = 1584230400000ms [2020-03-15], [1] = -777945600000ms [1945-05-08], [2] = -62135596800000ms [0001-01-01], [3] = -62135683200000ms [year <= 0], [4] = 123ms [non-multiple of 86400000]}')
    check_heap_repr(gdb_arrow, 'heap_time32_array_s', 'arrow::Time32Array of type arrow::time32(arrow::TimeUnit::SECOND), length 3, offset 0, null count 1 = {[0] = null, [1] = -123s, [2] = 456s}')
    check_heap_repr(gdb_arrow, 'heap_time32_array_ms', 'arrow::Time32Array of type arrow::time32(arrow::TimeUnit::MILLI), length 3, offset 0, null count 1 = {[0] = null, [1] = -123ms, [2] = 456ms}')
    check_heap_repr(gdb_arrow, 'heap_time64_array_us', 'arrow::Time64Array of type arrow::time64(arrow::TimeUnit::MICRO), length 3, offset 0, null count 1 = {[0] = null, [1] = -123us, [2] = 456us}')
    check_heap_repr(gdb_arrow, 'heap_time64_array_ns', 'arrow::Time64Array of type arrow::time64(arrow::TimeUnit::NANO), length 3, offset 0, null count 1 = {[0] = null, [1] = -123ns, [2] = 456ns}')
    check_heap_repr(gdb_arrow, 'heap_month_interval_array', 'arrow::MonthIntervalArray of length 3, offset 0, null count 1 = {[0] = 123M, [1] = -456M, [2] = null}')
    check_heap_repr(gdb_arrow, 'heap_day_time_interval_array', 'arrow::DayTimeIntervalArray of length 2, offset 0, null count 1 = {[0] = 1d-600ms, [1] = null}')
    check_heap_repr(gdb_arrow, 'heap_month_day_nano_interval_array', 'arrow::MonthDayNanoIntervalArray of length 2, offset 0, null count 1 = {[0] = 1M-600d5000ns, [1] = null}')
    check_heap_repr(gdb_arrow, 'heap_duration_array_s', 'arrow::DurationArray of type arrow::duration(arrow::TimeUnit::SECOND), length 2, offset 0, null count 1 = {[0] = null, [1] = -1234567890123456789s}')
    check_heap_repr(gdb_arrow, 'heap_duration_array_ns', 'arrow::DurationArray of type arrow::duration(arrow::TimeUnit::NANO), length 2, offset 0, null count 1 = {[0] = null, [1] = -1234567890123456789ns}')
    check_heap_repr(gdb_arrow, 'heap_timestamp_array_s', 'arrow::TimestampArray of type arrow::timestamp(arrow::TimeUnit::SECOND), length 4, offset 0, null count 1 = {[0] = null, [1] = 0s [1970-01-01 00:00:00], [2] = -2203932304s [1900-02-28 12:34:56], [3] = 63730281600s [3989-07-14 00:00:00]}')
    check_heap_repr(gdb_arrow, 'heap_timestamp_array_ms', 'arrow::TimestampArray of type arrow::timestamp(arrow::TimeUnit::MILLI), length 3, offset 0, null count 1 = {[0] = null, [1] = -2203932303877ms [1900-02-28 12:34:56.123], [2] = 63730281600789ms [3989-07-14 00:00:00.789]}')
    check_heap_repr(gdb_arrow, 'heap_timestamp_array_us', 'arrow::TimestampArray of type arrow::timestamp(arrow::TimeUnit::MICRO), length 3, offset 0, null count 1 = {[0] = null, [1] = -2203932303345679us [1900-02-28 12:34:56.654321], [2] = 63730281600456789us [3989-07-14 00:00:00.456789]}')
    check_heap_repr(gdb_arrow, 'heap_timestamp_array_ns', 'arrow::TimestampArray of type arrow::timestamp(arrow::TimeUnit::NANO), length 2, offset 0, null count 1 = {[0] = null, [1] = -2203932303012345679ns [1900-02-28 12:34:56.987654321]}')
    check_heap_repr(gdb_arrow, 'heap_decimal128_array', 'arrow::Decimal128Array of type arrow::decimal128(30, 6), length 3, offset 0, null count 1 = {[0] = null, [1] = -1234567890123456789.012345, [2] = 1234567890123456789.012345}')
    check_heap_repr(gdb_arrow, 'heap_decimal256_array', 'arrow::Decimal256Array of type arrow::decimal256(50, 6), length 2, offset 0, null count 1 = {[0] = null, [1] = -123456789012345678901234567890123456789.012345}')
    check_heap_repr(gdb_arrow, 'heap_decimal128_array_sliced', 'arrow::Decimal128Array of type arrow::decimal128(30, 6), length 1, offset 1, unknown null count = {[0] = -1234567890123456789.012345}')
    check_heap_repr(gdb_arrow, 'heap_fixed_size_binary_array', 'arrow::FixedSizeBinaryArray of type arrow::fixed_size_binary(3), length 3, offset 0, null count 1 = {[0] = null, [1] = "abc", [2] = "\\000\\037\\377"}')
    check_heap_repr(gdb_arrow, 'heap_fixed_size_binary_array_zero_width', 'arrow::FixedSizeBinaryArray of type arrow::fixed_size_binary(0), length 2, offset 0, null count 1 = {[0] = null, [1] = ""}')
    check_heap_repr(gdb_arrow, 'heap_fixed_size_binary_array_sliced', 'arrow::FixedSizeBinaryArray of type arrow::fixed_size_binary(3), length 1, offset 1, unknown null count = {[0] = "abc"}')
    check_heap_repr(gdb_arrow, 'heap_binary_array', 'arrow::BinaryArray of length 3, offset 0, null count 1 = {[0] = null, [1] = "abcd", [2] = "\\000\\037\\377"}')
    check_heap_repr(gdb_arrow, 'heap_large_binary_array', 'arrow::LargeBinaryArray of length 3, offset 0, null count 1 = {[0] = null, [1] = "abcd", [2] = "\\000\\037\\377"}')
    check_heap_repr(gdb_arrow, 'heap_string_array', 'arrow::StringArray of length 3, offset 0, null count 1 = {[0] = null, [1] = "héhé", [2] = "invalid \\\\xff char"}')
    check_heap_repr(gdb_arrow, 'heap_large_string_array', 'arrow::LargeStringArray of length 3, offset 0, null count 1 = {[0] = null, [1] = "héhé", [2] = "invalid \\\\xff char"}')
    check_heap_repr(gdb_arrow, 'heap_binary_array_sliced', 'arrow::BinaryArray of length 1, offset 1, unknown null count = {[0] = "abcd"}')
    check_heap_repr(gdb_arrow, 'heap_list_array', 'arrow::ListArray of type arrow::list(arrow::int64()), length 3, offset 0, null count 1')