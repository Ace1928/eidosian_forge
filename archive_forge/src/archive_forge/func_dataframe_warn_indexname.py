from traitlets import TraitError, TraitType
import numpy as np
import pandas as pd
import warnings
import datetime as dt
import six
def dataframe_warn_indexname(trait, value):
    if value.index.name is not None:
        warnings.warn("The '%s' dataframe trait of the %s instance disregards the index name" % (trait.name, trait.this_class))
        value = value.reset_index()
    return value