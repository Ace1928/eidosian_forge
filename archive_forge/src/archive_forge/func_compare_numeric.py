import doctest
import re
import decimal
def compare_numeric(self, want, got, optionflags):
    """
        Compares want and got by scanning for numbers. The numbers are
        compared using an epsilon extracted from optionflags. The text
        pieces between the numbers are compared falling back to the
        default implementation of OutputChecker.

        Returns a pair (status, data) where status is 'OK' if the
        comparison passed or indicates how it failed with data containing
        information that can be used to format the text explaining the
        differences.
        """
    split_want = re.split(number_re, want)
    split_got = re.split(number_re, got)
    if len(split_want) != len(split_got):
        return ('COUNT', (len(split_want) // number_split_stride, len(split_got) // number_split_stride))
    flags = optionflags | NUMERIC_DEFAULT_OPTIONFLAGS
    for i in range(0, len(split_want), number_split_stride):
        if not doctest.OutputChecker.check_output(self, split_want[i], split_got[i], flags):
            return ('TEXT', (split_want[i], split_got[i], i))
    epsilon = decimal.Decimal(0.1) ** get_precision(optionflags)
    rows = []
    max_diff = 0
    for i in range(1, len(split_want), number_split_stride):
        number_want = split_want[i]
        number_got = split_got[i]
        is_interval_want = bool(split_want[i + 2])
        is_interval_got = bool(split_got[i + 2])
        if is_interval_want != is_interval_got:
            return ('TYPE', (is_interval_want, number_got))
        decimal_want = to_decimal(split_want[i:i + number_group_count])
        decimal_got = to_decimal(split_got[i:i + number_group_count])
        diff = abs(decimal_want - decimal_got)
        failed = diff > epsilon
        max_diff = max(max_diff, diff)
        rows.append((number_want, number_got, failed, diff))
    if max_diff > epsilon:
        return ('NUMERIC', (rows, max_diff))
    return ('OK', None)