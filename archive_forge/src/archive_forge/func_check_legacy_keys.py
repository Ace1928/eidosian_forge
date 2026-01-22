from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, Union
from wasabi import msg
@root_validator(pre=True)
def check_legacy_keys(cls, obj: Dict[str, Any]) -> Dict[str, Any]:
    if 'spacy_version' in obj:
        msg.warn('Your project configuration file includes a `spacy_version` key, which is now deprecated. Weasel will not validate your version of spaCy.')
    if 'check_requirements' in obj:
        msg.warn('Your project configuration file includes a `check_requirements` key, which is now deprecated. Weasel will not validate your requirements.')
    return obj