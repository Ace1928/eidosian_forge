import re
def parse_pairwise(lines, results):
    """Parse results from pairwise comparisons."""
    pair_re = re.compile('\\d+ \\((.+)\\) ... \\d+ \\((.+)\\)')
    pairwise = {}
    seq1 = None
    seq2 = None
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        pair_res = pair_re.match(line)
        if pair_res:
            seq1 = pair_res.group(1)
            seq2 = pair_res.group(2)
            if seq1 not in pairwise:
                pairwise[seq1] = {}
            if seq2 not in pairwise:
                pairwise[seq2] = {}
        if len(line_floats) == 1 and seq1 is not None and (seq2 is not None):
            pairwise[seq1][seq2] = {'lnL': line_floats[0]}
            pairwise[seq2][seq1] = pairwise[seq1][seq2]
        elif len(line_floats) == 6 and seq1 is not None and (seq2 is not None):
            pairwise[seq1][seq2].update({'t': line_floats[0], 'S': line_floats[1], 'N': line_floats[2], 'omega': line_floats[3], 'dN': line_floats[4], 'dS': line_floats[5]})
            pairwise[seq2][seq1] = pairwise[seq1][seq2]
    if pairwise:
        results['pairwise'] = pairwise
    return results