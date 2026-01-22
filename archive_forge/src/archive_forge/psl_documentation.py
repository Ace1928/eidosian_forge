from itertools import chain
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq, reverse_complement, UndefinedSequenceError
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, ExactPosition, SimpleLocation, CompoundLocation
Return a string with a single alignment formatted as one PSL line.