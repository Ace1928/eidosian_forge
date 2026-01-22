from .coin_api import *
from .cplex_api import *
from .gurobi_api import *
from .glpk_api import *
from .choco_api import *
from .mipcl_api import *
from .mosek_api import *
from .scip_api import *
from .xpress_api import *
from .highs_api import *
from .copt_api import *
from .core import *
import json
def setConfigInformation(**keywords):
    """
    set the data in the configuration file
    at the moment will only edit things in [locations]
    the keyword value pairs come from the keywords dictionary
    """
    config = Parser()
    config.read(config_filename)
    for key, val in keywords.items():
        config.set('locations', key, val)
    fp = open(config_filename, 'w')
    config.write(fp)
    fp.close()