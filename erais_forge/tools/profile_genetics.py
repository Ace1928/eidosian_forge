import cProfile
import pstats
import io
import asyncio
import sys
from pathlib import Path

# Setup Path
FORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORGE_ROOT / "erais_forge/src"))
sys.path.insert(0, str(FORGE_ROOT / "lib"))

from erais_forge.models import Gene
from erais_forge.library import GeneticLibrary
from eidosian_core import eidosian

@eidosian(profile=True)
def heavy_genetic_op(lib: GeneticLibrary, count: int = 50):
    """Perform a batch of genetic operations for profiling."""
    for i in range(count):
        g = Gene(name=f"p_{i}", content=f"content_{i}", kind="prompt", fitness=0.5)
        lib.add_gene(g)
    return lib.get_top_genes(limit=10)

def main():
    print("--- 📉 ERAIS Genetics Profiling Session ---")
    benchmark_path = Path("profile_genetics.json")
    lib = GeneticLibrary(storage_path=benchmark_path)
    
    # Run the operation
    heavy_genetic_op(lib, count=50)
    
    # Cleanup
    if benchmark_path.exists():
        import os
        os.remove(benchmark_path)
    
    print("Profiling session complete. Detailed stats available in system trace logs.")

if __name__ == "__main__":
    main()
