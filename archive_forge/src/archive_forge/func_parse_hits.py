import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
def parse_hits(self):
    """Parse a HMMER2 hit block, beginning with the hit table."""
    hit_placeholders = []
    while self.read_next():
        if self.line.startswith('Parsed'):
            break
        if self.line.find('no hits') > -1:
            break
        if self.line.startswith('Sequence') or self.line.startswith('Model') or self.line.startswith('-------- '):
            continue
        fields = self.line.split()
        id_ = fields.pop(0)
        domain_obs_num = int(fields.pop())
        evalue = float(fields.pop())
        bitscore = float(fields.pop())
        description = ' '.join(fields).strip()
        hit = _HitPlaceholder()
        hit.id_ = id_
        hit.evalue = evalue
        hit.bitscore = bitscore
        hit.description = description
        hit.domain_obs_num = domain_obs_num
        hit_placeholders.append(hit)
    return hit_placeholders