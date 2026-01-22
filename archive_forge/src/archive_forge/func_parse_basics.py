import re
def parse_basics(lines, results):
    """Parse the basic information that should be present in most codeml output files."""
    multi_models = False
    multi_genes = False
    version_re = re.compile('.+ \\(in paml version (\\d+\\.\\d+[a-z]*).*')
    model_re = re.compile('Model:\\s+(.+)')
    num_genes_re = re.compile('\\(([0-9]+) genes: separate data\\)')
    codon_freq_re = re.compile('Codon frequenc[a-z\\s]{3,7}:\\s+(.+)')
    siteclass_re = re.compile('Site-class models:\\s*([^\\s]*)')
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        version_res = version_re.match(line)
        if version_res is not None:
            results['version'] = version_res.group(1)
            continue
        model_res = model_re.match(line)
        if model_res is not None:
            results['model'] = model_res.group(1)
        num_genes_res = num_genes_re.search(line)
        if num_genes_res is not None:
            results['genes'] = []
            num_genes = int(num_genes_res.group(1))
            for n in range(num_genes):
                results['genes'].append({})
            multi_genes = True
            continue
        codon_freq_res = codon_freq_re.match(line)
        if codon_freq_res is not None:
            results['codon model'] = codon_freq_res.group(1)
            continue
        siteclass_res = siteclass_re.match(line)
        if siteclass_res is not None:
            siteclass_model = siteclass_res.group(1)
            if siteclass_model != '':
                results['site-class model'] = siteclass_model
                multi_models = False
            else:
                multi_models = True
        if 'ln Lmax' in line and line_floats:
            results['lnL max'] = line_floats[0]
    return (results, multi_models, multi_genes)