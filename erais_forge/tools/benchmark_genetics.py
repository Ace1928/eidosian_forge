import time
import sys
import os
from pathlib import Path
from typing import List

# Setup Path
FORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORGE_ROOT / "erais_forge/src"))

from erais_forge.models import Gene
from erais_forge.library import GeneticLibrary

def run_benchmark(gene_count: int = 100):
    print(f"--- 🧬 ERAIS Genetics Benchmark ({gene_count} genes) ---")
    
    # Use a temporary benchmark file
    benchmark_path = Path("benchmark_genetics.json")
    lib = GeneticLibrary(storage_path=benchmark_path)
    
    # 1. Benchmark Write
    start_write = time.time()
    for i in range(gene_count):
        gene = Gene(name=f"gene_{i}", content=f"content_{i}", kind="prompt")
        lib.add_gene(gene)
    end_write = time.time()
    write_time = (end_write - start_write) * 1000
    print(f"Write Throughput: {write_time:.2f}ms total ({write_time/gene_count:.2f}ms/gene)")
    
    # 2. Benchmark Read/Retrieval
    start_read = time.time()
    top = lib.get_top_genes(limit=10)
    end_read = time.time()
    read_time = (end_read - start_read) * 1000
    print(f"Retrieval Latency (Top 10): {read_time:.2f}ms")
    
    # Cleanup
    if benchmark_path.exists():
        os.remove(benchmark_path)
    
    # Validation
    if write_time/gene_count < 5.0:
        print("RESULT: [PASSED] Meets performance spec (< 5ms/gene)")
    else:
        print("RESULT: [FAILED] Performance regression detected.")

if __name__ == "__main__":
    run_benchmark(100)
