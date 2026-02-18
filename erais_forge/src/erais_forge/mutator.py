from typing import Optional
from .models import Gene
from llm_forge.core.manager import ModelManager
from eidosian_core import eidosian

class GeneMutator:
    """Uses LLMs to perform intelligent crossover and mutation on genes."""

    def __init__(self, llm: Optional[ModelManager] = None):
        self.llm = llm or ModelManager()

    @eidosian()
    async def mutate(self, gene: Gene, instructions: str = "Optimize for clarity and precision.") -> Gene:
        """Propose a mutated version of a gene."""
        prompt = f"""You are the Eidosian Gene Mutator. 
Original Gene ({gene.kind}):
{gene.content}

Instructions: {instructions}

Propose a mutated version of this gene that improves its effectiveness while maintaining its core intent. 
Return ONLY the new content.
"""
        response = self.llm.generate(prompt=prompt)
        new_content = response.text or gene.content
        
        return Gene(
            name=f"{gene.name}_mutation",
            content=new_content,
            kind=gene.kind,
            generation=gene.generation + 1,
            parent_ids=[gene.id],
            metadata={"mutation_instructions": instructions}
        )
