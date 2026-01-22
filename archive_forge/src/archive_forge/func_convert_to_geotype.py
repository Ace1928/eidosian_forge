from holoviews.core import Element
from holoviews.operation.element import contours
from holoviews.operation.stats import bivariate_kde
from .. import element as gv_element
from ..element import _Element
from .projection import ( # noqa (API import)
from .resample import resample_geometry # noqa (API import)
def convert_to_geotype(element, crs=None):
    """
    Converts a HoloViews element type to the equivalent GeoViews
    element if given a coordinate reference system.
    """
    geotype = getattr(gv_element, type(element).__name__, None)
    if crs is None or geotype is None or isinstance(element, _Element):
        return element
    return element.clone(new_type=geotype, crs=crs)