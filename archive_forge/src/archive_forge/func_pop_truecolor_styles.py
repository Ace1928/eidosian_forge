from typing import Dict, List
def pop_truecolor_styles() -> Dict[str, Rule]:
    global truecolor_rules
    styles = dict([(item.klass.strip('.'), item) for item in truecolor_rules])
    truecolor_rules = []
    return styles