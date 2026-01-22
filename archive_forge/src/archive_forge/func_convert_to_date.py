from .core import *
from .helpers import DelimitedList, any_open_tag, any_close_tag
from datetime import datetime
@staticmethod
def convert_to_date(fmt: str='%Y-%m-%d'):
    """
        Helper to create a parse action for converting parsed date string to Python datetime.date

        Params -
        - fmt - format to be passed to datetime.strptime (default= ``"%Y-%m-%d"``)

        Example::

            date_expr = pyparsing_common.iso8601_date.copy()
            date_expr.set_parse_action(pyparsing_common.convert_to_date())
            print(date_expr.parse_string("1999-12-31"))

        prints::

            [datetime.date(1999, 12, 31)]
        """

    def cvt_fn(ss, ll, tt):
        try:
            return datetime.strptime(tt[0], fmt).date()
        except ValueError as ve:
            raise ParseException(ss, ll, str(ve))
    return cvt_fn