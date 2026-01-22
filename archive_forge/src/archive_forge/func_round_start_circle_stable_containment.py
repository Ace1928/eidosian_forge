from math import copysign, cos, hypot, isclose, pi
from fontTools.misc.roundTools import otRound
def round_start_circle_stable_containment(c0, r0, c1, r1):
    """Round start circle so that it stays inside/outside end circle after rounding.

    The rounding of circle coordinates to integers may cause an abrupt change
    if the start circle c0 is so close to the end circle c1's perimiter that
    it ends up falling outside (or inside) as a result of the rounding.
    To keep the gradient unchanged, we nudge it in the right direction.

    See:
    https://github.com/googlefonts/colr-gradients-spec/issues/204
    https://github.com/googlefonts/picosvg/issues/158
    """
    start, end = (Circle(c0, r0), Circle(c1, r1))
    inside_before_round = start.inside(end)
    round_start = start.round()
    round_end = end.round()
    inside_after_round = round_start.inside(round_end)
    if inside_before_round == inside_after_round:
        return round_start
    elif inside_after_round:
        direction = _vector_between(round_end.centre, round_start.centre)
        radius_delta = +1.0
    else:
        direction = _vector_between(round_start.centre, round_end.centre)
        radius_delta = -1.0
    dx, dy = _rounding_offset(direction)
    max_attempts = 2
    for _ in range(max_attempts):
        if round_start.concentric(round_end):
            round_start.radius += radius_delta
            assert round_start.radius >= 0
        else:
            round_start.move(dx, dy)
        if inside_before_round == round_start.inside(round_end):
            break
    else:
        raise AssertionError(f'Rounding circle {start} {('inside' if inside_before_round else 'outside')} {end} failed after {max_attempts} attempts!')
    return round_start