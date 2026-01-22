from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def parse_offer(cls, offer):
    """
        Parse an offer into its component parts.

        :param offer: A media type or range in the format
                      ``type/subtype[;params]``.
        :return: A named tuple containing ``(*type*, *subtype*, *params*)``.

                 | *params* is a list containing ``(*parameter name*, *value*)``
                   values.

        :raises ValueError: If the offer does not match the required format.

        """
    if isinstance(offer, AcceptOffer):
        return offer
    match = cls.media_type_compiled_re.match(offer)
    if not match:
        raise ValueError('Invalid value for an Accept offer.')
    groups = match.groups()
    offer_type, offer_subtype = groups[0].split('/')
    offer_params = cls._parse_media_type_params(media_type_params_segment=groups[1])
    if offer_type == '*' or offer_subtype == '*':
        raise ValueError('Invalid value for an Accept offer.')
    return AcceptOffer(offer_type.lower(), offer_subtype.lower(), tuple(((name.lower(), value) for name, value in offer_params)))