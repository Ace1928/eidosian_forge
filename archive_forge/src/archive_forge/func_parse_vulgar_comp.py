import re
from ._base import _BaseExonerateParser, _BaseExonerateIndexer, _STRAND_MAP
from typing import Type
def parse_vulgar_comp(hsp, vulgar_comp):
    """Parse the vulgar components present in the hsp dictionary."""
    qstarts = [hsp['query_start']]
    qends = []
    hstarts = [hsp['hit_start']]
    hends = []
    hsp['query_split_codons'] = []
    hsp['hit_split_codons'] = []
    hsp['query_ner_ranges'] = []
    hsp['hit_ner_ranges'] = []
    qpos = hsp['query_start']
    hpos = hsp['hit_start']
    qmove = 1 if hsp['query_strand'] >= 0 else -1
    hmove = 1 if hsp['hit_strand'] >= 0 else -1
    vcomps = re.findall(_RE_VCOMP, vulgar_comp)
    for idx, match in enumerate(vcomps):
        label, qstep, hstep = (match[0], int(match[1]), int(match[2]))
        assert label in 'MCGF53INS', 'Unexpected vulgar label: %r' % label
        if label in 'MCGS':
            if vcomps[idx - 1][0] not in 'MCGS':
                qstarts.append(qpos)
                hstarts.append(hpos)
        if label == 'S':
            qstart, hstart = (qpos, hpos)
            qend = qstart + qstep * qmove
            hend = hstart + hstep * hmove
            sqstart, sqend = (min(qstart, qend), max(qstart, qend))
            shstart, shend = (min(hstart, hend), max(hstart, hend))
            hsp['query_split_codons'].append((sqstart, sqend))
            hsp['hit_split_codons'].append((shstart, shend))
        qpos += qstep * qmove
        hpos += hstep * hmove
        if idx == len(vcomps) - 1 or (label in 'MCGS' and vcomps[idx + 1][0] not in 'MCGS'):
            qends.append(qpos)
            hends.append(hpos)
    for seq_type in ('query_', 'hit_'):
        strand = hsp[seq_type + 'strand']
        if strand < 0:
            hsp[seq_type + 'start'], hsp[seq_type + 'end'] = (hsp[seq_type + 'end'], hsp[seq_type + 'start'])
            if seq_type == 'query_':
                qstarts, qends = (qends, qstarts)
            else:
                hstarts, hends = (hends, hstarts)
    hsp['query_ranges'] = list(zip(qstarts, qends))
    hsp['hit_ranges'] = list(zip(hstarts, hends))
    return hsp