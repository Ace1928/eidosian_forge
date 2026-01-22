import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
For the intrinsic function get_attr when getting all attributes.

        :returns: a dict of all of the resource's attribute values, excluding
                  the "show" attribute.
        