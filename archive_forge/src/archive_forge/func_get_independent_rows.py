from ..pari import pari
import fractions
def get_independent_rows(rows, explain_rows, desired_determinant=None, sort_rows_key=None):
    row_explain_pairs = list(zip(rows, explain_rows))
    if sort_rows_key:
        row_explain_pairs.sort(key=lambda row_explain_pair: sort_rows_key(row_explain_pair[1]))
    result = _get_independent_rows_recursive(row_explain_pairs, len(rows[0]), desired_determinant, [], [])
    if not result:
        raise Exception('Could not find enough independent rows')
    return result