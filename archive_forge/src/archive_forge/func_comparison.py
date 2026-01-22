import colorsys
import io
from time import process_time
from pip._vendor.rich import box
from pip._vendor.rich.color import Color
from pip._vendor.rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from pip._vendor.rich.markdown import Markdown
from pip._vendor.rich.measure import Measurement
from pip._vendor.rich.pretty import Pretty
from pip._vendor.rich.segment import Segment
from pip._vendor.rich.style import Style
from pip._vendor.rich.syntax import Syntax
from pip._vendor.rich.table import Table
from pip._vendor.rich.text import Text
def comparison(renderable1: RenderableType, renderable2: RenderableType) -> Table:
    table = Table(show_header=False, pad_edge=False, box=None, expand=True)
    table.add_column('1', ratio=1)
    table.add_column('2', ratio=1)
    table.add_row(renderable1, renderable2)
    return table