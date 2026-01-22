import re
from Bio.SearchIO._utils import read_forward, removesuffix
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
Iterate over Hmmer3TextIndexer; yields query results' key, offsets, 0.