from . import matrix
from . import homology
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVariety import PtolemyVariety
from .utilities import MethodMappingList
def compute_solutions(self, *args, **kwargs):
    return MethodMappingList([p.compute_solutions(*args, **kwargs) for p in self])