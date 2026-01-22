from pyomo.common.tempfiles import TempfileManager
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.writers import LPWriter
import logging
import math
from pyomo.common.collections import ComponentMap
from typing import Optional, Sequence, NoReturn, List, Mapping, Dict
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.timing import HierarchicalTimer
import sys
import time
from pyomo.common.log import LogStream
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.errors import PyomoException
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.core.staleflag import StaleFlagManager
def remove_variables(self, variables: List[_GeneralVarData]):
    self._writer.remove_variables(variables)