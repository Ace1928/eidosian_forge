from typing import Optional
from docutils.writers.latex2e import Babel
def get_mainlanguage_options(self) -> Optional[str]:
    """Return options for polyglossia's ``\\setmainlanguage``."""
    if self.use_polyglossia is False:
        return None
    elif self.language == 'german':
        language = super().language_name(self.language_code)
        if language == 'ngerman':
            return 'spelling=new'
        else:
            return 'spelling=old'
    else:
        return None