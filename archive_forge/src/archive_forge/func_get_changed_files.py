import difflib
from pathlib import Path
from typing import Dict, Iterable, Tuple
from parso import split_lines
from jedi.api.exceptions import RefactoringError
from jedi.inference.value.namespace import ImplicitNSName
def get_changed_files(self) -> Dict[Path, ChangedFile]:

    def calculate_to_path(p):
        if p is None:
            return p
        p = str(p)
        for from_, to in renames:
            if p.startswith(str(from_)):
                p = str(to) + p[len(str(from_)):]
        return Path(p)
    renames = self.get_renames()
    return {path: ChangedFile(self._inference_state, from_path=path, to_path=calculate_to_path(path), module_node=next(iter(map_)).get_root_node(), node_to_str_map=map_) for path, map_ in sorted(self._file_to_node_changes.items(), key=lambda x: x[0] or Path(''))}