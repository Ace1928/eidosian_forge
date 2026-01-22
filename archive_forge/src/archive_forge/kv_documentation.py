import functools
import re
from ovs.flow.decoders import decode_default
Parse the key-value pairs in string.

        The input string is assumed to contain a list of comma (or space)
        separated key-value pairs.

        Key-values pairs can have multiple different delimiters, eg:
            "key1:value1,key2=value2,key3(value3)".

        Also, we can stumble upon a "free" keywords, e.g:
            "key1=value1,key2=value2,free_keyword".
        We consider this as keys without a value.

        So, to parse the string we do the following until the end of the
        string is found:

            1 - Skip any leading comma's or spaces.
            2 - Find the next delimiter (or end_of_string character).
            3 - Depending on the delimiter, obtain the key and the value.
                For instance, if the delimiter is "(", find the next matching
                ")".
            4 - Use the KVDecoders to decode the key-value.
            5 - Store the KeyValue object with the corresponding metadata.

        Raises:
            ParseError if any parsing error occurs.
        