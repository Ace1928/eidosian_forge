import time
import shutil
import random
from pathlib import Path
from memory_forge.backends.json_store import JsonBackend
from memory_forge.backends.chroma_store import ChromaBackend
from memory_forge.core.interfaces import MemoryItem

def benchmark_backend(backend, name, n=1000):
    print(f"\n--- Benchmarking {name} (N={n}) ---")
    
    # Write
    start = time.perf_counter()
    for i in range(n):
        embedding = [random.random() for _ in range(10)]
        backend.add(MemoryItem(f"Memory {i}", embedding=embedding))
    write_time = time.perf_counter() - start
    print(f"Write: {n/write_time:.2f} ops/sec")
    
    # Read
    start = time.perf_counter()
    for i in range(n):
        backend.get(f"Memory {i}") # IDs are usually UUIDs, but we didn't force ID in loop. 
        # Actually MemoryItem generates UUID by default.
        # Let's fix the loop to store IDs.
    
    # Corrected Loop
    backend.clear()
    ids = []
    start = time.perf_counter()
    for i in range(n):
        embedding = [random.random() for _ in range(10)]
        item = MemoryItem(f"Memory {i}", embedding=embedding)
        backend.add(item)
        ids.append(item.id)
    write_time = time.perf_counter() - start
    print(f"Write (Corrected): {n/write_time:.2f} ops/sec")

    start = time.perf_counter()
    for mid in ids:
        backend.get(mid)
    read_time = time.perf_counter() - start
    print(f"Read: {n/read_time:.2f} ops/sec")
    
    # Search
    start = time.perf_counter()
    query = [random.random() for _ in range(10)]
    for _ in range(100): # 100 searches
        backend.search(query, limit=5)
    search_time = time.perf_counter() - start
    print(f"Search: {100/search_time:.2f} ops/sec")

def main():
    tmp = Path("./bench_data")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    
    # JSON
    jb = JsonBackend(str(tmp / "bench.json"))
    benchmark_backend(jb, "JSON Backend", n=100) # Small N for JSON
    
    # Chroma
    cb = ChromaBackend("bench", str(tmp / "chroma"))
    benchmark_backend(cb, "Chroma Backend", n=100)
    
    shutil.rmtree(tmp)

if __name__ == "__main__":
    main()
