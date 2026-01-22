from .arc import EllipticalArc
import re
def parse_path(pathdef, pen, current_pos=(0, 0), arc_class=EllipticalArc):
    """Parse SVG path definition (i.e. "d" attribute of <path> elements)
    and call a 'pen' object's moveTo, lineTo, curveTo, qCurveTo and closePath
    methods.

    If 'current_pos' (2-float tuple) is provided, the initial moveTo will
    be relative to that instead being absolute.

    If the pen has an "arcTo" method, it is called with the original values
    of the elliptical arc curve commands:

        pen.arcTo(rx, ry, rotation, arc_large, arc_sweep, (x, y))

    Otherwise, the arcs are approximated by series of cubic Bezier segments
    ("curveTo"), one every 90 degrees.
    """
    current_pos = complex(*current_pos)
    elements = list(_tokenize_path(pathdef))
    elements.reverse()
    start_pos = None
    command = None
    last_control = None
    have_arcTo = hasattr(pen, 'arcTo')
    while elements:
        if elements[-1] in COMMANDS:
            last_command = command
            command = elements.pop()
            absolute = command in UPPERCASE
            command = command.upper()
        else:
            if command is None:
                raise ValueError('Unallowed implicit command in %s, position %s' % (pathdef, len(pathdef.split()) - len(elements)))
            last_command = command
        if command == 'M':
            x = elements.pop()
            y = elements.pop()
            pos = float(x) + float(y) * 1j
            if absolute:
                current_pos = pos
            else:
                current_pos += pos
            if start_pos is not None:
                pen.endPath()
            pen.moveTo((current_pos.real, current_pos.imag))
            start_pos = current_pos
            command = 'L'
        elif command == 'Z':
            if current_pos != start_pos:
                pen.lineTo((start_pos.real, start_pos.imag))
            pen.closePath()
            current_pos = start_pos
            start_pos = None
            command = None
        elif command == 'L':
            x = elements.pop()
            y = elements.pop()
            pos = float(x) + float(y) * 1j
            if not absolute:
                pos += current_pos
            pen.lineTo((pos.real, pos.imag))
            current_pos = pos
        elif command == 'H':
            x = elements.pop()
            pos = float(x) + current_pos.imag * 1j
            if not absolute:
                pos += current_pos.real
            pen.lineTo((pos.real, pos.imag))
            current_pos = pos
        elif command == 'V':
            y = elements.pop()
            pos = current_pos.real + float(y) * 1j
            if not absolute:
                pos += current_pos.imag * 1j
            pen.lineTo((pos.real, pos.imag))
            current_pos = pos
        elif command == 'C':
            control1 = float(elements.pop()) + float(elements.pop()) * 1j
            control2 = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j
            if not absolute:
                control1 += current_pos
                control2 += current_pos
                end += current_pos
            pen.curveTo((control1.real, control1.imag), (control2.real, control2.imag), (end.real, end.imag))
            current_pos = end
            last_control = control2
        elif command == 'S':
            if last_command not in 'CS':
                control1 = current_pos
            else:
                control1 = current_pos + current_pos - last_control
            control2 = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j
            if not absolute:
                control2 += current_pos
                end += current_pos
            pen.curveTo((control1.real, control1.imag), (control2.real, control2.imag), (end.real, end.imag))
            current_pos = end
            last_control = control2
        elif command == 'Q':
            control = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j
            if not absolute:
                control += current_pos
                end += current_pos
            pen.qCurveTo((control.real, control.imag), (end.real, end.imag))
            current_pos = end
            last_control = control
        elif command == 'T':
            if last_command not in 'QT':
                control = current_pos
            else:
                control = current_pos + current_pos - last_control
            end = float(elements.pop()) + float(elements.pop()) * 1j
            if not absolute:
                end += current_pos
            pen.qCurveTo((control.real, control.imag), (end.real, end.imag))
            current_pos = end
            last_control = control
        elif command == 'A':
            rx = abs(float(elements.pop()))
            ry = abs(float(elements.pop()))
            rotation = float(elements.pop())
            arc_large = bool(int(elements.pop()))
            arc_sweep = bool(int(elements.pop()))
            end = float(elements.pop()) + float(elements.pop()) * 1j
            if not absolute:
                end += current_pos
            if have_arcTo:
                pen.arcTo(rx, ry, rotation, arc_large, arc_sweep, (end.real, end.imag))
            else:
                arc = arc_class(current_pos, rx, ry, rotation, arc_large, arc_sweep, end)
                arc.draw(pen)
            current_pos = end
    if start_pos is not None:
        pen.endPath()