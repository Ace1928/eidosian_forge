from blib2to3.pytree import Leaf
def normalize_numeric_literal(leaf: Leaf) -> None:
    """Normalizes numeric (float, int, and complex) literals.

    All letters used in the representation are normalized to lowercase."""
    text = leaf.value.lower()
    if text.startswith(('0o', '0b')):
        pass
    elif text.startswith('0x'):
        text = format_hex(text)
    elif 'e' in text:
        text = format_scientific_notation(text)
    elif text.endswith('j'):
        text = format_complex_number(text)
    else:
        text = format_float_or_int_string(text)
    leaf.value = text