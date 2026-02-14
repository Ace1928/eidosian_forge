# Main Orchestrator Module
from typing import List
import numpy as np
import logging
from fractal_foundation import FractalIntelligence, ProcessingMode
from emotional_substrate import EmotionalSubstrate, EmotionalState
from chaotic_sandbox import ChaoticSandbox


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MainOrchestrator:
    """Orchestrates the integration of Fractal Intelligence, Emotional Substrate, and Chaotic Sandbox."""
    
    def __init__(self, layers: int, chaos_intensity: float, mutation_rate: float):
        self.fractal_engine = FractalIntelligence(
            layers=layers,
            chaos_intensity=chaos_intensity,
            mode=ProcessingMode.QUANTUM,  # Use quantum mode for enhanced processing
            parallel=True,
            feedback_enabled=True
        )
        self.emotional_system = EmotionalSubstrate(
            EmotionalState.CURIOSITY,
            adaptability=0.2  # Increased adaptability for more dynamic responses
        )
        self.chaotic_sandbox = ChaoticSandbox(
            dimensions=3, 
            chaos_intensity=chaos_intensity, 
            mutation_rate=mutation_rate
        )

    def run(self, initial_data: List[float], iterations: int):
        """Executes the unified system for a defined number of iterations."""
        data = initial_data.copy()
        results_history = []
        
        for i in range(iterations):
            logger.info(f"Iteration {i} starting.")

            # Emotional influence on initial data
            data = [self.emotional_system.influence_decision(value) for value in data]

            # Process through Fractal Intelligence
            fractal_output = self.fractal_engine.run(data)

            # Apply Chaotic Sandbox for destabilization and mutation
            chaos_results = self.chaotic_sandbox.simulate(iterations=1)[0]
            chaotic_influence = np.mean(chaos_results['state'])
            
            # Combine fractal and chaos results
            processed_output = [v * (1 + chaotic_influence) for v in fractal_output]

            # Update Emotional Substrate based on comprehensive feedback
            feedback = np.mean(processed_output) + chaotic_influence
            self.emotional_system.update_emotional_state(feedback)

            # Store iteration results
            results_history.append({
                'iteration': i,
                'fractal_output': fractal_output,
                'chaos_influence': chaotic_influence,
                'emotional_state': self.emotional_system.current_state.state.name,
                'final_output': processed_output
            })

            # Log detailed iteration results
            logger.info(
                f"Iteration {i} completed:\n"
                f"  Fractal Output: {fractal_output}\n"
                f"  Chaos Influence: {chaotic_influence:.3f}\n"
                f"  Emotional State: {self.emotional_system.current_state.state.name}\n"
                f"  Final Output: {processed_output}"
            )
            
            data = processed_output

        return results_history

# Integration Example
if __name__ == "__main__":
    orchestrator = MainOrchestrator(
        layers=5, 
        chaos_intensity=0.3, 
        mutation_rate=0.2
    )
    
    # Test with sample sequence
    initial_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = orchestrator.run(initial_data, iterations=10)
    
    # Analyze and display results
    print("\nProcessing Results Summary:")
    print("-" * 50)
    for result in results:
        print(f"\nIteration {result['iteration']}:")
        print(f"  Input -> Output Scale: {np.mean(result['final_output'])/np.mean(initial_data):.2f}x")
        print(f"  Emotional State: {result['emotional_state']}")
        print(f"  Chaos Factor: {result['chaos_influence']:.3f}")
