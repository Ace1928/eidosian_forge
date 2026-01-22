import re
import sys
def text_analyse(text):
    try:
        rv = f(text)
    except Exception:
        return 0.0
    if not rv:
        return 0.0
    try:
        return min(1.0, max(0.0, float(rv)))
    except (ValueError, TypeError):
        return 0.0