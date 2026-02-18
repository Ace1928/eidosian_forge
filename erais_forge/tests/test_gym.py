import pytest
import asyncio
from erais_forge.gym import FitnessGym
from erais_forge.models import Genome, Gene

@pytest.mark.asyncio
async def test_gym_evaluation_interface():
    gym = FitnessGym()
    gene = Gene(name="test", content="logic", kind="code")
    genome = Genome(genes=[gene])
    
    # Test valid environment
    fitness = await gym.evaluate_genome(genome, environment="gene_particles")
    assert 0.0 <= fitness <= 1.0
    
    # Test invalid environment
    with pytest.raises(ValueError):
        await gym.evaluate_genome(genome, environment="invalid_env")

@pytest.mark.asyncio
async def test_gym_chess_evaluation():
    gym = FitnessGym()
    gene = Gene(name="test_chess", content="logic", kind="code")
    genome = Genome(genes=[gene])
    fitness = await gym.evaluate_genome(genome, environment="agentic_chess")
    assert fitness > 0
