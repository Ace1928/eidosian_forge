import os
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from . import Des
from . import Cla
from . import Hie
from . import Residues
from Bio import SeqIO
from Bio.Seq import Seq
def parse_domain(term):
    """Convert an ASTRAL header string into a Scop domain.

    An ASTRAL (http://astral.stanford.edu/) header contains a concise
    description of a SCOP domain. A very similar format is used when a
    Domain object is converted into a string.  The Domain returned by this
    method contains most of the SCOP information, but it will not be located
    within the SCOP hierarchy (i.e. The parent node will be None). The
    description is composed of the SCOP protein and species descriptions.

    A typical ASTRAL header looks like --
    >d1tpt_1 a.46.2.1 (1-70) Thymidine phosphorylase {Escherichia coli}
    """
    m = _domain_re.match(term)
    if not m:
        raise ValueError('Domain: ' + term)
    dom = Domain()
    dom.sid = m.group(1)
    dom.sccs = m.group(2)
    dom.residues = Residues.Residues(m.group(3))
    if not dom.residues.pdbid:
        dom.residues.pdbid = dom.sid[1:5]
    dom.description = m.group(4).strip()
    return dom