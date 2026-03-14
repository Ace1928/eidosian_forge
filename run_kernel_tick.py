import sys
from pathlib import Path

# Add project root and src to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "agent_forge/src"))

from agent_forge.consciousness.kernel import ConsciousnessKernel

state_dir = ROOT / "state"

print("Instantiating new kernel to force module reload...")
kernel = ConsciousnessKernel(state_dir)
print(f"Active modules: {[m.name for m in kernel.modules]}")

print("Running 5 ticks to generate potential motor events...")
for i in range(5):
    res = kernel.tick()
    print(f"Tick {i} executed, emitted events: {res.emitted_events}")

ledger_file = state_dir / "ledger" / "continuity_ledger.jsonl"
if ledger_file.exists():
    lines = ledger_file.read_text().strip().splitlines()
    print("\n--- Last 3 Ledger Entries ---")
    for line in lines[-3:]:
        print(line)
