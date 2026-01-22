from . import mf2_classes
from .dom_helpers import get_attr, get_children, get_img, get_textContent, try_urljoin
def get_url_child(children):
    """take a list of children and finds a valid child for url property"""
    poss_as = [c for c in children if c.name == 'a']
    if len(poss_as) == 1:
        poss_a = poss_as[0]
        if not mf2_classes.root(poss_a.get('class', []), filtered_roots):
            return poss_a
    poss_areas = [c for c in children if c.name == 'area']
    if len(poss_areas) == 1:
        poss_area = poss_areas[0]
        if not mf2_classes.root(poss_area.get('class', []), filtered_roots):
            return poss_area