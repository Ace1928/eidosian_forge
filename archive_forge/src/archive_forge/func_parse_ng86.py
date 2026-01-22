import re
def parse_ng86(lines, results):
    """Parse the Nei & Gojobori (1986) section of the results.

    Nei_Gojobori results are organized in a lower
    triangular matrix, with the sequence names labeling
    the rows and statistics in the format:
    w (dN dS) per column
    Example row (2 columns):
    0.0000 (0.0000 0.0207) 0.0000 (0.0000 0.0421)
    """
    sequences = []
    for line in lines:
        matrix_row_res = re.match('^([^\\s]+?)(\\s+-?\\d+\\.\\d+.*$|\\s*$|-1.0000\\s*\\(.*$)', line)
        if matrix_row_res is not None:
            line_floats_res = re.findall('-*\\d+\\.\\d+', matrix_row_res.group(2))
            line_floats = [float(val) for val in line_floats_res]
            seq_name = matrix_row_res.group(1).strip()
            sequences.append(seq_name)
            results[seq_name] = {}
            for i in range(0, len(line_floats), 3):
                NG86 = {}
                NG86['omega'] = line_floats[i]
                NG86['dN'] = line_floats[i + 1]
                NG86['dS'] = line_floats[i + 2]
                results[seq_name][sequences[i // 3]] = {'NG86': NG86}
                results[sequences[i // 3]][seq_name] = {'NG86': NG86}
    return (results, sequences)