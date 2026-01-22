import breezy
import breezy.commands
import breezy.help
from breezy.doc_generate import get_autodoc_datetime
def infogen(options, outfile):
    d = get_autodoc_datetime()
    params = {'brzcmd': options.brz_name, 'datestamp': d.strftime('%Y-%m-%d'), 'timestamp': d.strftime('%Y-%m-%d %H:%M:%S +0000'), 'version': breezy.__version__}
    outfile.write(preamble % params)