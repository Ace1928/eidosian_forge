from __future__ import annotations
import asyncio
from typing import List, Optional
from .models import Gene, Genome, Generation
from .library import GeneticLibrary
from .mutator import GeneMutator
from eidosian_core import eidosian

class EvolutionaryController:
    """The orchestrator of the ERAIS evolutionary loop."""

    def __init__(self, library: Optional[GeneticLibrary] = None, mutator: Optional[GeneMutator] = None):
        self.library = library or GeneticLibrary()
        self.mutator = mutator or GeneMutator()

    @eidosian()
    async def evolve_generation(self, kind: str, population_size: int = 5) -> Generation:
        """Run one evolutionary cycle for a specific kind of gene."""
        # 1. Selection: Get the best existing genes
        top_genes = self.library.get_top_genes(kind=kind, limit=population_size)
        
        # If no genes exist, we can't evolve yet.
        if not top_genes:
            raise ValueError(f"No genes of kind '{kind}' found in library to evolve.")

        # 2. Mutation: Propose variations
        new_genes = []
        for parent in top_genes:
            mutation = await self.mutator.mutate(parent)
            self.library.add_gene(mutation)
            new_genes.append(mutation)

        # 3. Aggregation: Form a new generation
        # (In a real scenario, we would evaluate fitness here before forming genomes)
        genomes = [Genome(genes=[g]) for g in new_genes]
        
        gen_id = 1
        # Simple generation ID logic for now
        # ... logic to determine next generation ID ...

        generation = Generation(
            id=gen_id,
            genomes=genomes,
            avg_fitness=sum(g.fitness for g in new_genes) / len(new_genes) if new_genes else 0.0,
            best_fitness=max(g.fitness for g in new_genes) if new_genes else 0.0
        )
        
        return generation

    @eidosian()
    async def run_loop(self, kind: str, max_generations: int = 1):
        """Execute the RSI loop for multiple generations."""
        for i in range(max_generations):
            print(f"Evolving Generation {i+1}...")
            await self.evolve_generation(kind)
