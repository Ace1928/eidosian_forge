from typing import Callable, Dict, Iterable, Union
from Bio.Align import MultipleSeqAlignment
from Bio.File import as_handle
from Bio.SeqIO import AbiIO
from Bio.SeqIO import AceIO
from Bio.SeqIO import FastaIO
from Bio.SeqIO import GckIO
from Bio.SeqIO import IgIO  # IntelliGenetics or MASE format
from Bio.SeqIO import InsdcIO  # EMBL and GenBank
from Bio.SeqIO import NibIO
from Bio.SeqIO import PdbIO
from Bio.SeqIO import PhdIO
from Bio.SeqIO import PirIO
from Bio.SeqIO import QualityIO  # FastQ and qual files
from Bio.SeqIO import SeqXmlIO
from Bio.SeqIO import SffIO
from Bio.SeqIO import SnapGeneIO
from Bio.SeqIO import SwissIO
from Bio.SeqIO import TabIO
from Bio.SeqIO import TwoBitIO
from Bio.SeqIO import UniprotIO
from Bio.SeqIO import XdnaIO
from Bio.SeqRecord import SeqRecord
from .Interfaces import _TextIOSource
def over_ride(record):
    """Over-ride molecule in-place."""
    record.annotations['molecule_type'] = molecule_type
    return record