from a single quote by the algorithm. Therefore, a text like::
import re, sys
def educateQuotes(text, language='en'):
    """
    Parameter:  - text string (unicode or bytes).
                - language (`BCP 47` language tag.)
    Returns:    The `text`, with "educated" curly quote characters.

    Example input:  "Isn't this fun?"
    Example output: “Isn’t this fun?“;
    """
    smart = smartchars(language)
    punct_class = '[!"#\\$\\%\'()*+,-.\\/:;<=>?\\@\\[\\\\\\]\\^_`{|}~]'
    close_class = '[^\\ \\t\\r\\n\\[\\{\\(\\-]'
    open_class = '[\u200b\u200c]'
    dec_dashes = '&#8211;|&#8212;'
    text = re.sub("^'(?=%s\\\\B)" % (punct_class,), smart.csquote, text)
    text = re.sub('^"(?=%s\\\\B)' % (punct_class,), smart.cpquote, text)
    text = re.sub('"\'(?=\\w)', smart.opquote + smart.osquote, text)
    text = re.sub('\'"(?=\\w)', smart.osquote + smart.opquote, text)
    if language.startswith('en'):
        text = re.sub("'(?=\\d{2}s)", smart.apostrophe, text)
    opening_single_quotes_regex = re.compile("\n                    (# ?<=  # look behind fails: requires fixed-width pattern\n                            \\s          |   # a whitespace char, or\n                            %s          |   # another separating char, or\n                            &nbsp;      |   # a non-breaking space entity, or\n                            [–—]        |   # literal dashes, or\n                            --          |   # dumb dashes, or\n                            &[mn]dash;  |   # dash entities (named or\n                            %s          |   # decimal or\n                            &\\#x201[34];    # hex)\n                    )\n                    '                 # the quote\n                    (?=\\w)            # followed by a word character\n                    " % (open_class, dec_dashes), re.VERBOSE | re.UNICODE)
    text = opening_single_quotes_regex.sub('\\1' + smart.osquote, text)
    if smart.csquote != smart.apostrophe:
        apostrophe_regex = re.compile("(?<=(\\w|\\d))'(?=\\w)", re.UNICODE)
        text = apostrophe_regex.sub(smart.apostrophe, text)
    closing_single_quotes_regex = re.compile("\n                    (?<=%s)\n                    '\n                    " % close_class, re.VERBOSE)
    text = closing_single_quotes_regex.sub(smart.csquote, text)
    text = re.sub("'", smart.osquote, text)
    opening_double_quotes_regex = re.compile('\n                    (\n                            \\s          |   # a whitespace char, or\n                            %s          |   # another separating char, or\n                            &nbsp;      |   # a non-breaking space entity, or\n                            [–—]        |   # literal dashes, or\n                            --          |   # dumb dashes, or\n                            &[mn]dash;  |   # dash entities (named or\n                            %s          |   # decimal or\n                            &\\#x201[34];    # hex)\n                    )\n                    "                 # the quote\n                    (?=\\w)            # followed by a word character\n                    ' % (open_class, dec_dashes), re.VERBOSE | re.UNICODE)
    text = opening_double_quotes_regex.sub('\\1' + smart.opquote, text)
    closing_double_quotes_regex = re.compile('\n                    (\n                    (?<=%s)" | # char indicating the quote should be closing\n                    "(?=\\s)    # whitespace behind\n                    )\n                    ' % (close_class,), re.VERBOSE | re.UNICODE)
    text = closing_double_quotes_regex.sub(smart.cpquote, text)
    text = re.sub('"', smart.opquote, text)
    return text