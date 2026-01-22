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
def getAstralDomainsFromSQL(self, column):
    """Load ASTRAL domains from the MySQL database.

        Load a set of astral domains from a column in the astral table of a MYSQL
        database (which can be created with writeToSQL(...).
        """
    cur = self.db_handle.cursor()
    cur.execute('SELECT sid FROM astral WHERE ' + column + '=1')
    data = cur.fetchall()
    data = [self.scop.getDomainBySid(x[0]) for x in data]
    return data