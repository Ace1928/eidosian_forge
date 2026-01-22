from traitlets import Unicode, Bool, validate, TraitError
from .trait_types import datetime_serialization, Datetime, naive_serialization
from .valuewidget import ValueWidget
from .widget import register
from .widget_core import CoreWidget
from .widget_description import DescriptionWidget
Enforce min <= value <= max