import re
from . import value_class_pattern
from .datetime_helpers import DATETIME_RE, TIME_RE, normalize_datetime
from .dom_helpers import get_attr, get_img, get_textContent, try_urljoin
Process e-* properties