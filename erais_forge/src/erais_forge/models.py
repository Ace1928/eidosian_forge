from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class Gene(BaseModel):
    """A single unit of evolution (Prompt or Code snippet)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    content: str
    kind: str  # 'prompt' or 'code'
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)

class Genome(BaseModel):
    """A collection of Genes forming a complete personality or toolset."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    genes: List[Gene]
    overall_fitness: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)

class Generation(BaseModel):
    """A snapshot of the population at a point in time."""
    id: int
    genomes: List[Genome]
    avg_fitness: float
    best_fitness: float
    timestamp: datetime = Field(default_factory=datetime.now)
