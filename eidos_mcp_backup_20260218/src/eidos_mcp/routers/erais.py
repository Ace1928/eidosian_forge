from __future__ import annotations
import json
from typing import List, Optional
from ..core import tool
from erais_forge.controller import EvolutionaryController
from erais_forge.library import GeneticLibrary
from eidosian_core import eidosian

# Initialize ERAIS components
lib = GeneticLibrary()
controller = EvolutionaryController(library=lib)

@tool(
    name="erais_evolve",
    description="Run an evolutionary cycle for a specific gene type.",
    parameters={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "'prompt' or 'code'"},
            "generations": {"type": "integer", "default": 1},
            "population_size": {"type": "integer", "default": 5}
        },
        "required": ["kind"],
    },
)
@eidosian()
async def erais_evolve(kind: str, generations: int = 1, population_size: int = 5) -> str:
    """Run an evolutionary cycle."""
    try:
        results = []
        for i in range(generations):
            gen = await controller.evolve_generation(kind, population_size=population_size)
            results.append({
                "generation": gen.id,
                "avg_fitness": gen.avg_fitness,
                "best_fitness": gen.best_fitness
            })
        return json.dumps({"status": "success", "results": results}, indent=2)
    except Exception as e:
        return f"Error during evolution: {e}"

@tool(
    name="erais_list_genes",
    description="List top-performing genes from the genetic library.",
    parameters={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "'prompt' or 'code'"},
            "limit": {"type": "integer", "default": 10}
        }
    },
)
@eidosian()
def erais_list_genes(kind: Optional[str] = None, limit: int = 10) -> str:
    """List top genes."""
    genes = lib.get_top_genes(kind=kind, limit=limit)
    out = []
    for g in genes:
        out.append(f"[{g.id[:8]}] {g.name} (Fitness: {g.fitness}, Gen: {g.generation})")
    return "\n".join(out) if out else "No genes found matching criteria."
