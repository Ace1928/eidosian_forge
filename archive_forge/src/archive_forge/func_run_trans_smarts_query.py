import os
import re
import sqlite3
import subprocess
import sys
from optparse import OptionParser
from indexing import cansmirk, heavy_atom_count
from rfrag import fragment_mol
def run_trans_smarts_query(transform):
    lhs, rhs = transform.split('>>')
    matching_lhs = []
    matching_rhs = []
    os.environ['RD_USESQLLITE'] = '1'
    cmd = "python $RDBASE/Projects/DbCLI/SearchDb.py --dbDir=%s_smarts --smarts='%s' --silent" % (pre, lhs)
    p1 = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = p1.communicate()[0].decode().rstrip()
    matching_lhs = output.split('\n')
    cmd = "python $RDBASE/Projects/DbCLI/SearchDb.py --dbDir=%s_smarts --smarts='%s' --silent" % (pre, rhs)
    p1 = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = p1.communicate()[0].decode().rstrip()
    matching_rhs = output.split('\n')
    lhs_q_string = "','".join(matching_lhs)
    lhs_q_string = "'%s'" % lhs_q_string
    rhs_q_string = "','".join(matching_rhs)
    rhs_q_string = "'%s'" % rhs_q_string
    query_sql = '\n    select  lhs_smi.smiles,\n            lhs.cmpd_id,\n            lhs.core_smi,\n            rhs_smi.smiles,\n            rhs.cmpd_id,\n            rhs.core_smi,\n            context_table.context_smi\n    from    (select cmpd_id,core_smi,context_id from core_table where core_smi_ni in (%s) ) lhs,\n            (select cmpd_id,core_smi,context_id from core_table where core_smi_ni in (%s) ) rhs,\n            cmpd_smisp lhs_smi,\n            cmpd_smisp rhs_smi,\n            context_table\n    where   lhs.context_id = rhs.context_id\n            and context_table.context_id = rhs.context_id\n            and lhs_smi.cmpd_id = lhs.cmpd_id\n            and rhs_smi.cmpd_id = rhs.cmpd_id\n            and lhs.cmpd_id != rhs.cmpd_id\n            and rhs_smi.cmpd_size-context_table.context_size <= %s\n            and lhs_smi.cmpd_size-context_table.context_size <= %s ' % (lhs_q_string, rhs_q_string, max_size, max_size)
    cursor.execute(query_sql)
    results = cursor.fetchall()
    for r in results:
        smirks, context = cansmirk(str(r[2]), str(r[5]), str(r[6]))
        if have_id:
            print('%s,%s,%s,%s,%s,%s,%s,%s' % (transform, id, r[0], r[3], r[1], r[4], smirks, context))
        else:
            print('%s,%s,%s,%s,%s,%s,%s' % (transform, r[0], r[3], r[1], r[4], smirks, context))