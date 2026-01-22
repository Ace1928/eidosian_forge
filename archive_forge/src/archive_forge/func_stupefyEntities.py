from a single quote by the algorithm. Therefore, a text like::
import re, sys
def stupefyEntities(text, language='en'):
    """
    Parameter:  String (unicode or bytes).
    Returns:    The `text`, with each SmartyPants character translated to
                its ASCII counterpart.

    Example input:  “Hello — world.”
    Example output: "Hello -- world."
    """
    smart = smartchars(language)
    text = re.sub(smart.endash, '-', text)
    text = re.sub(smart.emdash, '--', text)
    text = re.sub(smart.osquote, "'", text)
    text = re.sub(smart.csquote, "'", text)
    text = re.sub(smart.opquote, '"', text)
    text = re.sub(smart.cpquote, '"', text)
    text = re.sub(smart.ellipsis, '...', text)
    return text