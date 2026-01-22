from typing import Any, Optional
from langchain_core.runnables.graph import Graph, LabelsDict
def get_node_label(self, label: str) -> str:
    label = self.labels.get('nodes', {}).get(label, label)
    return f'<<B>{label}</B>>'