import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp

        :param section: The section to add the docs to.

        :param value: The input / output values representing the parameters that
                      are included in the example.

        :param comments: The dictionary containing all the comments to be
                         applied to the example.

        :param path: A list describing where the documenter is in traversing the
                     parameters. This is used to find the equivalent location
                     in the comments dictionary.
        