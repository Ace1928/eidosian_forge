import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
def deactivate():
    global original_converter
    if original_converter is None:
        return
    conversion.set_conversion(original_converter)
    original_converter = None