import spherogram
import string
def regina_DT(code):
    """
    >>> L = regina_DT('flmnopHKRQGabcdeJI')
    >>> L
    <Link: 1 comp; 18 cross>
    >>> L = regina_DT('hPONrMjsLfiQGDCBKae')
    >>> L
    <Link: 1 comp; 19 cross>
    """
    cross = len(code)
    l = string.ascii_letters[cross - 1]
    dt_prefix = 'DT: ' + l + 'a' + l
    return spherogram.Link(dt_prefix + code)