from pathlib import Path
from agent_forge.autonomy.supervisor import AutonomySupervisor
from agent_forge.consciousness.kernel import ConsciousnessKernel

def test_imports():
    state_dir = Path("./data/runtime/test_autonomy")
    state_dir.mkdir(parents=True, exist_ok=True)
    
    print("Testing AutonomySupervisor initialization...")
    supervisor = AutonomySupervisor(state_dir=state_dir, repo_root=".")
    print("Success.")
    
    print("Testing ConsciousnessKernel initialization...")
    kernel = ConsciousnessKernel(state_dir=state_dir)
    print("Success.")
    
    print("Testing homeostatic drives...")
    pulse = {"cpu": {"percent": 10}, "memory": {"percent": 20}, "disk": {"percent": 30}}
    health = kernel.runtime_health()
    drives = supervisor.homeostasis.compute_drives(pulse, health)
    print(f"Drives: {drives}")
    assert hasattr(drives, "curiosity_drive")
    assert hasattr(drives, "caution_drive")
    
    print("Testing ledger heartbeat...")
    history = kernel.ledger.get_history()
    print(f"Ledger history length: {len(history)}")
    assert len(history) > 0

if __name__ == "__main__":
    try:
        test_imports()
        print("All Phase 2 sanity checks passed.")
    except Exception as e:
        print(f"Sanity check failed: {e}")
        import traceback
        traceback.print_exc()
