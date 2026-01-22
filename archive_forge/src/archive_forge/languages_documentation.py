from string import ascii_letters
from typing import List, Optional
Metadata about a language useful for training models

    :ivar name: The human name for the language, in English.
    :type name: str
    :ivar iso_code: 2-letter ISO 639-1 if possible, 3-letter ISO code otherwise,
                    or use another catalog as a last resort.
    :type iso_code: str
    :ivar use_ascii: Whether or not ASCII letters should be included in trained
                     models.
    :type use_ascii: bool
    :ivar charsets: The charsets we want to support and create data for.
    :type charsets: list of str
    :ivar alphabet: The characters in the language's alphabet. If `use_ascii` is
                    `True`, you only need to add those not in the ASCII set.
    :type alphabet: str
    :ivar wiki_start_pages: The Wikipedia pages to start from if we're crawling
                            Wikipedia for training data.
    :type wiki_start_pages: list of str
    