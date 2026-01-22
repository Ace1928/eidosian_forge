import patiencediff
from merge3 import Merge3
from ... import merge
from .parser import simple_parse_lines
def merge_text(self, params):
    """Perform a simple 3-way merge of a bzr NEWS file.

        Each section of a bzr NEWS file is essentially an ordered set of bullet
        points, so we can simply take a set of bullet points, determine which
        bullets to add and which to remove, sort, and reserialize.
        """
    this_lines = list(simple_parse_lines(params.this_lines))
    other_lines = list(simple_parse_lines(params.other_lines))
    base_lines = list(simple_parse_lines(params.base_lines))
    m3 = Merge3(base_lines, this_lines, other_lines, sequence_matcher=patiencediff.PatienceSequenceMatcher)
    result_chunks = []
    for group in m3.merge_groups():
        if group[0] == 'conflict':
            _, base, a, b = group
            for line_set in [base, a, b]:
                for line in line_set:
                    if line[0] != 'bullet':
                        return ('not_applicable', None)
            new_in_a = set(a).difference(base)
            new_in_b = set(b).difference(base)
            all_new = new_in_a.union(new_in_b)
            deleted_in_a = set(base).difference(a)
            deleted_in_b = set(base).difference(b)
            final = all_new.difference(deleted_in_a).difference(deleted_in_b)
            final = sorted(final, key=sort_key)
            result_chunks.extend(final)
        else:
            result_chunks.extend(group[1])
    result_lines = '\n\n'.join((chunk[1] for chunk in result_chunks))
    return ('success', result_lines)