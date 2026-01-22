from statsmodels.compat.python import lzip
from io import StringIO
import numpy as np
from statsmodels.iolib import SimpleTable
def hypothesis_test_table(results, title, null_hyp):
    fmt = dict(_default_table_fmt, data_fmts=['%#15.6F', '%#15.6F', '%#15.3F', '%s'])
    buf = StringIO()
    table = SimpleTable([[results['statistic'], results['crit_value'], results['pvalue'], str(results['df'])]], ['Test statistic', 'Critical Value', 'p-value', 'df'], [''], title=None, txt_fmt=fmt)
    buf.write(title + '\n')
    buf.write(str(table) + '\n')
    buf.write(null_hyp + '\n')
    buf.write('Conclusion: %s H_0' % results['conclusion'])
    buf.write(' at %.2f%% significance level' % (results['signif'] * 100))
    return buf.getvalue()