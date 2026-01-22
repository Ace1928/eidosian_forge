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
def getAscendentFromSQL(self, node, type):
    """Get ascendents using SQL backend."""
    if nodeCodeOrder.index(type) >= nodeCodeOrder.index(node.type):
        return None
    cur = self.db_handle.cursor()
    cur.execute('SELECT ' + type + ' from cla WHERE ' + node.type + '=%s', node.sunid)
    result = cur.fetchone()
    if result is not None:
        return self.getNodeBySunid(result[0])
    else:
        return None