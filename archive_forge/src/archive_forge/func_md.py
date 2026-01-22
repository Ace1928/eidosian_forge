from markdownify import MarkdownConverter
from bs4 import BeautifulSoup
def md(html, **options):
    return ImageBlockConverter(**options).convert(html)