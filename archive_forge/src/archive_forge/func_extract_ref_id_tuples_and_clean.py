import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import (
from langchain_core.pydantic_v1 import root_validator
def extract_ref_id_tuples_and_clean(self, contents: List[str]) -> List[Tuple[str, int]]:
    """
        Extracts reference IDs from the contents and cleans up the ref tags.

        This function processes a list of strings, searching for reference ID tags
        at the beginning and end of each string. When a ref tag is found, it is
        removed from the string, and its ID is recorded. If no ref ID is found,
        a generic ID of "-1" is assigned.

        The search for ref tags is performed only at the beginning and
        end of the string, with the assumption that there will
        be at most one ref ID per string. Malformed ref tags are
        handled gracefully.

        Args:
            contents (List[str]): A list of contents to be processed.

        Returns:
            List[Tuple[str, int]]: The cleaned string and the associated ref ID.

        Examples:
            >>> strings_list = [
                    '<#ref0#> Example content <#ref0#>',
                    'Content with no ref ID.'
                ]
            >>> extract_ref_id_tuples_and_clean(strings_list)
            [('Example content', 0), ('Content with no ref ID.', -1)]
        """
    ref_id_tuples = []
    for content in contents:
        clean_string = content.strip()
        if not clean_string:
            continue
        ref_id = None
        for pattern in [self._pattern_beginning, self._pattern_ending]:
            match = pattern.search(clean_string)
            if match:
                ref_id = match.group(1)
                clean_string = pattern.sub('', clean_string).strip()
        ref_id_to_use = int(ref_id) if ref_id and ref_id.isdigit() else -1
        ref_id_tuples.append((clean_string, ref_id_to_use))
    return ref_id_tuples