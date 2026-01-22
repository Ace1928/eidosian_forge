import os
import re
import sqlite3
import subprocess
import sys
from optparse import OptionParser
from indexing import cansmirk, heavy_atom_count
from rfrag import fragment_mol
def run_mmp_query(cmpd_id, cmpd_size):
    query_sql = "\n    select  c.cmpd_id,\n            c.core_smi,\n            con.context_smi,\n            con.context_size\n    from    core_table c, context_table con\n    where   c.context_id in (select context_id from core_table where cmpd_id = '%s')\n            and c.context_id = con.context_id" % cmpd_id
    cursor.execute(query_sql)
    results = cursor.fetchall()
    print_smallest_change_mmp(results, cmpd_id, cmpd_size)