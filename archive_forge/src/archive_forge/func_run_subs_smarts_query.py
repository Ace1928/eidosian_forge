import os
import re
import sqlite3
import subprocess
import sys
from optparse import OptionParser
from indexing import cansmirk, heavy_atom_count
from rfrag import fragment_mol
def run_subs_smarts_query(subs_smarts):
    os.environ['RD_USESQLLITE'] = '1'
    temp_core_ni_file = 'temp_core_ni_file_%s' % os.getpid()
    cmd = "python $RDBASE/Projects/DbCLI/SearchDb.py --dbDir=%s_smarts --smarts='%s' --silent >%s" % (pre, subs_smarts, temp_core_ni_file)
    subprocess.Popen(cmd, shell=True).wait()
    infile = open(temp_core_ni_file, 'r')
    for row in infile:
        row = row.rstrip()
        query_sql = "\n        select  lhs_smi.smiles,\n                lhs.cmpd_id,\n                lhs.core_smi,\n                rhs_smi.smiles,\n                rhs.cmpd_id,\n                rhs.core_smi,\n                context_table.context_smi,\n                rhs_smi.cmpd_size-context_table.context_size\n        from    (select cmpd_id,core_smi,context_id from core_table where core_smi_ni = '%s') lhs,\n                core_table rhs,\n                cmpd_smisp lhs_smi,\n                cmpd_smisp rhs_smi,\n                context_table\n        where   lhs.context_id = rhs.context_id\n                and context_table.context_id = rhs.context_id\n                and lhs_smi.cmpd_id = lhs.cmpd_id\n                and rhs_smi.cmpd_id = rhs.cmpd_id\n                and lhs.cmpd_id != rhs.cmpd_id\n                and rhs_smi.cmpd_size-context_table.context_size <= %s\n                and lhs_smi.cmpd_size-context_table.context_size <= %s" % (row, max_size, max_size)
        cursor.execute(query_sql)
        results = cursor.fetchall()
        for r in results:
            smirks, context = cansmirk(str(r[2]), str(r[5]), str(r[6]))
            if have_id:
                print('%s,%s,%s,%s,%s,%s,%s' % (id, r[0], r[3], r[1], r[4], smirks, context))
            else:
                print('%s,%s,%s,%s,%s,%s' % (r[0], r[3], r[1], r[4], smirks, context))
    infile.close()
    os.unlink(temp_core_ni_file)