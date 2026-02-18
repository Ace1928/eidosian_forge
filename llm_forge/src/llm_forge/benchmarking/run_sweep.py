import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# Setup Path
FORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORGE_ROOT / "llm_forge/src"))

from llm_forge.benchmarking.engine_bench import Benchmarker, BenchResult

async def run_sweep(model_path: str, output_path: str):
    bench = Benchmarker()
    
    # Define sweep parameters
    prompt_lengths = [128, 512, 1024]
    gen_lengths = [32, 128, 512]
    
    print(f"--- 📊 Starting LLM Performance Sweep ---")
    print(f"Model: {model_path}")
    
    results = await bench.run_throughput_test(
        model_path=model_path,
        p_tokens=prompt_lengths,
        g_tokens=gen_lengths
    )
    
    if not results:
        print("❌ Error: Benchmark failed or binary not found.")
        return

    # Aggregate and Save
    data = [r.model_dump() for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Sweep complete. Results saved to {output_path}")
    
    # Print Summary
    print("\n--- Summary ---")
    for r in results:
        print(f"P={r.n_prompt:4} G={r.n_gen:4} | P_TPS: {r.tokens_sec_prompt:6.2f} | G_TPS: {r.tokens_sec_gen:6.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="bench_results.json")
    args = parser.parse_args()
    
    asyncio.run(run_sweep(args.model, args.output))
