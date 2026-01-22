import mplfinance as mpf
import pandas as pd
import textwrap
def left_formatter(value):
    if not isinstance(value, str):
        return f'{value:<}'
    elif value[0:maxwidth] == '-' * maxwidth:
        return f'{value:<{w}.{w}s}'
    elif len(value) > maxwidth:
        return f'{value:<{wm3}.{wm3}s}...'
    else:
        return f'{value:<{w}.{w}s}'