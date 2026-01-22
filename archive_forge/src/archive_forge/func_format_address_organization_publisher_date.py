from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_address_organization_publisher_date(self, e, include_organization=True):
    """Format address, organization, publisher, and date.
        Everything is optional, except the date.
        """
    if include_organization:
        organization = optional_field('organization')
    else:
        organization = None
    return first_of[optional[join(sep=' ')[sentence[field('address'), date], sentence[organization, optional_field('publisher')]],], sentence[organization, optional_field('publisher'), date]]