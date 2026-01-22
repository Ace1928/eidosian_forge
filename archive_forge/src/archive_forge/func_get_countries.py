import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def get_countries():
    c1 = set([(e.alpha2, _c(e.name)) for e in pycountry.countries])
    ret = c1.union(country_additions + fuzzy_countrynames)
    return ret