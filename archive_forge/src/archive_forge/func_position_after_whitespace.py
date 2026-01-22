import json
from ..error import GraphQLSyntaxError
def position_after_whitespace(body, start_position):
    """Reads from body starting at start_position until it finds a
    non-whitespace or commented character, then returns the position of
    that character for lexing."""
    body_length = len(body)
    position = start_position
    while position < body_length:
        code = char_code_at(body, position)
        if code in ignored_whitespace_characters:
            position += 1
        elif code == 35:
            position += 1
            while position < body_length:
                code = char_code_at(body, position)
                if not (code is not None and (code > 31 or code == 9) and (code not in (10, 13))):
                    break
                position += 1
        else:
            break
    return position