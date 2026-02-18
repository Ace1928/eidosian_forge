import json
import os
from pathlib import Path
from typing import List, Optional, Dict
from .models import Gene, Genome
from eidosian_core import eidosian

class GeneticLibrary:
    """Persistent storage for Eidosian genes."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".eidosian/genetics/library.json"
        self.genes: Dict[str, Gene] = {}
        self.load()

    def load(self):
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for g_id, g_data in data.items():
                    self.genes[g_id] = Gene(**g_data)
            except (json.JSONDecodeError, Exception):
                self.genes = {}

    def save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {g_id: g.model_dump(mode='json') for g_id, g in self.genes.items()}
        self.storage_path.write_text(json.dumps(data, indent=2))

    @eidosian()
    def add_gene(self, gene: Gene):
        self.genes[gene.id] = gene
        self.save()

    @eidosian()
    def get_top_genes(self, kind: Optional[str] = None, limit: int = 10) -> List[Gene]:
        """Return the highest fitness genes."""
        filtered = [g for g in self.genes.values() if kind is None or g.kind == kind]
        sorted_genes = sorted(filtered, key=lambda x: x.fitness, reverse=True)
        return sorted_genes[:limit]

    @eidosian()
    def get_gene(self, gene_id: str) -> Optional[Gene]:
        return self.genes.get(gene_id)
