"""
Eidosian PyParticles V6 - Force System Tests

Comprehensive tests for all force types and the force registry.
"""

import pytest
import numpy as np
from pyparticles.physics.forces.base import (
    DropoffType, ForceDefinition,
    dropoff_linear, dropoff_inverse_square, dropoff_yukawa,
    dropoff_lennard_jones, dropoff_morse, dropoff_gaussian,
)
from pyparticles.physics.forces.potentials import (
    linear_force, inverse_force, inverse_square_force, inverse_cube_force,
    yukawa_force, lennard_jones_force, morse_force, gaussian_force,
    compute_force_unified,
)
from pyparticles.physics.forces.registry import (
    ForceRegistry, ForceType, create_preset_registry,
)


class TestDropoffFunctions:
    """Test individual dropoff/decay functions."""
    
    def test_dropoff_linear(self):
        params = np.array([0.0], dtype=np.float32)
        # At r=0, should be 1.0
        assert dropoff_linear(0.0, 1.0, params) == pytest.approx(1.0)
        # At r=r_max, should be 0.0
        assert dropoff_linear(1.0, 1.0, params) == pytest.approx(0.0)
        # At r=0.5*r_max, should be 0.5
        assert dropoff_linear(0.5, 1.0, params) == pytest.approx(0.5)
        # Beyond r_max, should be 0.0
        assert dropoff_linear(1.5, 1.0, params) == pytest.approx(0.0)
    
    def test_dropoff_inverse_square(self):
        params = np.array([0.01], dtype=np.float32)  # softening
        # Should decrease with distance
        f1 = dropoff_inverse_square(0.1, 1.0, params)
        f2 = dropoff_inverse_square(0.2, 1.0, params)
        assert f1 > f2
        # Beyond cutoff
        assert dropoff_inverse_square(1.5, 1.0, params) == 0.0
    
    def test_dropoff_yukawa(self):
        params = np.array([0.1, 0.01], dtype=np.float32)  # decay_length, softening
        f1 = dropoff_yukawa(0.05, 1.0, params)
        f2 = dropoff_yukawa(0.2, 1.0, params)
        assert f1 > f2  # Decays with distance
    
    def test_dropoff_lennard_jones(self):
        params = np.array([0.05, 0.005], dtype=np.float32)  # sigma, softening
        # Near sigma, should be near zero (equilibrium)
        # Far from sigma, should be small
        f_close = dropoff_lennard_jones(0.03, 0.5, params)  # Repulsive region
        f_far = dropoff_lennard_jones(0.1, 0.5, params)  # Attractive region
        # Close range should be positive (repulsive), far range negative (attractive)
        assert f_close > 0
    
    def test_dropoff_morse(self):
        params = np.array([0.1, 5.0], dtype=np.float32)  # r0, well_width
        # At equilibrium r0, force should be ~0
        f_eq = dropoff_morse(0.1, 0.5, params)
        assert abs(f_eq) < 0.01  # Near zero at equilibrium
        # Inside equilibrium: repulsive
        f_close = dropoff_morse(0.05, 0.5, params)
        assert f_close > 0
        # Outside equilibrium: attractive
        f_far = dropoff_morse(0.15, 0.5, params)
        assert f_far < 0
    
    def test_dropoff_gaussian(self):
        params = np.array([0.1], dtype=np.float32)  # sigma
        f1 = dropoff_gaussian(0.0, 1.0, params)
        f2 = dropoff_gaussian(0.2, 1.0, params)
        # Gaussian peaks at center
        assert f1 > f2


class TestForcePotentials:
    """Test force potential functions."""
    
    def test_linear_force_repulsion(self):
        # Inside min_r should be repulsive
        f = linear_force(dist=0.01, min_r=0.02, max_r=0.1, factor=1.0, strength=1.0)
        assert f < 0  # Repulsive (factor doesn't matter inside core)
    
    def test_linear_force_attraction(self):
        # Between min_r and max_r with positive factor should be attractive
        f = linear_force(dist=0.05, min_r=0.02, max_r=0.1, factor=1.0, strength=1.0)
        assert f > 0  # Attractive
    
    def test_linear_force_repulsion_factor(self):
        # Negative factor should invert the bell curve
        f = linear_force(dist=0.05, min_r=0.02, max_r=0.1, factor=-1.0, strength=1.0)
        assert f < 0  # Repulsive with negative factor
    
    def test_inverse_square_force(self):
        f = inverse_square_force(dist=0.1, max_r=0.5, factor=1.0, strength=1.0, softening=0.01)
        assert f > 0  # Positive factor = attractive
    
    def test_yukawa_force(self):
        f = yukawa_force(dist=0.1, max_r=0.5, factor=1.0, strength=1.0, 
                         decay_length=0.1, softening=0.01)
        assert f > 0  # Should be positive
        # Test decay
        f_close = yukawa_force(dist=0.05, max_r=0.5, factor=1.0, strength=1.0,
                               decay_length=0.1, softening=0.01)
        assert f_close > f  # Closer = stronger
    
    def test_lennard_jones_equilibrium(self):
        # At r = 2^(1/6) * sigma ≈ 1.122 * sigma, force should be near zero
        # But with softening, the equilibrium shifts slightly
        sigma = 0.05
        softening = 0.001
        # Test that force changes sign around equilibrium
        f_close = lennard_jones_force(dist=sigma * 0.8, max_r=0.5, factor=1.0, strength=1.0,
                                      sigma=sigma, softening=softening)
        f_far = lennard_jones_force(dist=sigma * 2.0, max_r=0.5, factor=1.0, strength=1.0,
                                    sigma=sigma, softening=softening)
        # Close range should be more repulsive (larger positive) than far range
        assert f_close > f_far
    
    def test_morse_equilibrium(self):
        r0 = 0.1
        f = morse_force(dist=r0, max_r=0.5, factor=1.0, strength=1.0,
                        r0=r0, well_width=5.0)
        assert abs(f) < 0.01  # Near zero at equilibrium
    
    def test_gaussian_force(self):
        f = gaussian_force(dist=0.05, max_r=0.5, factor=1.0, strength=1.0, sigma=0.1)
        assert f > 0


class TestComputeForceUnified:
    """Test the unified force dispatcher."""
    
    def test_linear_type(self):
        params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        f = compute_force_unified(
            dist=0.05, min_r=0.02, max_r=0.1,
            factor=1.0, strength=1.0, force_type=0,
            params=params
        )
        assert f != 0.0
    
    def test_inverse_square_type(self):
        params = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        f = compute_force_unified(
            dist=0.1, min_r=0.01, max_r=0.5,
            factor=1.0, strength=1.0, force_type=1,
            params=params
        )
        assert f > 0
    
    def test_yukawa_type(self):
        params = np.array([0.01, 0.1, 0.0], dtype=np.float32)  # softening, decay_length
        f = compute_force_unified(
            dist=0.1, min_r=0.01, max_r=0.5,
            factor=1.0, strength=1.0, force_type=5,
            params=params
        )
        assert f > 0
    
    def test_lennard_jones_type(self):
        params = np.array([0.005, 0.05, 0.0], dtype=np.float32)  # softening, sigma
        f = compute_force_unified(
            dist=0.03, min_r=0.01, max_r=0.3,
            factor=1.0, strength=1.0, force_type=6,
            params=params
        )
        # Close range should be repulsive (positive)
        assert f > 0
    
    def test_cutoff_respected(self):
        params = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        # All force types should return 0 beyond max_r
        for ftype in range(10):
            f = compute_force_unified(
                dist=1.0, min_r=0.01, max_r=0.5,
                factor=1.0, strength=1.0, force_type=ftype,
                params=params
            )
            assert f == 0.0, f"Force type {ftype} didn't respect cutoff"


class TestForceDefinition:
    """Test ForceDefinition dataclass."""
    
    def test_creation(self):
        mat = np.random.randn(4, 4).astype(np.float32)
        fd = ForceDefinition(
            name="Test",
            dropoff=DropoffType.LINEAR,
            matrix=mat,
            max_radius=0.1,
            min_radius=0.02,
            strength=1.5,
        )
        assert fd.name == "Test"
        assert fd.enabled == True
        assert fd.strength == 1.5
    
    def test_get_interaction(self):
        mat = np.array([[1.0, 0.5], [-0.5, 1.0]], dtype=np.float32)
        fd = ForceDefinition(
            name="Test", dropoff=DropoffType.LINEAR, matrix=mat,
            max_radius=0.1, min_radius=0.02,
        )
        assert fd.get_interaction(0, 1) == 0.5
        assert fd.get_interaction(1, 0) == -0.5
    
    def test_set_interaction_symmetric(self):
        mat = np.zeros((3, 3), dtype=np.float32)
        fd = ForceDefinition(
            name="Test", dropoff=DropoffType.LINEAR, matrix=mat,
            max_radius=0.1, min_radius=0.02, symmetric=True,
        )
        fd.set_interaction(0, 1, 0.75)
        assert fd.matrix[0, 1] == 0.75
        assert fd.matrix[1, 0] == 0.75  # Symmetrized
    
    def test_randomize_matrix(self):
        mat = np.zeros((4, 4), dtype=np.float32)
        fd = ForceDefinition(
            name="Test", dropoff=DropoffType.LINEAR, matrix=mat,
            max_radius=0.1, min_radius=0.02,
        )
        fd.randomize_matrix(-1.0, 1.0)
        assert not np.allclose(fd.matrix, 0.0)


class TestForceRegistry:
    """Test ForceRegistry class."""
    
    def test_creation_default(self):
        registry = ForceRegistry(num_types=4)
        assert len(registry.forces) > 0
        assert registry.num_types == 4
    
    def test_add_force(self):
        # Start with default forces and add one more
        registry = ForceRegistry(num_types=4)
        initial_count = len(registry.forces)
        mat = np.ones((4, 4), dtype=np.float32)
        idx = registry.add_force(ForceDefinition(
            name="Custom", dropoff=DropoffType.GAUSSIAN, matrix=mat,
            max_radius=0.2, min_radius=0.01,
        ))
        assert idx == initial_count  # Index is count before adding
        assert len(registry.forces) == initial_count + 1
    
    def test_enable_disable(self):
        registry = ForceRegistry(num_types=4)
        registry.enable_force("Particle Life", False)
        f = registry.get_force("Particle Life")
        assert f.enabled == False
    
    def test_get_active_forces(self):
        registry = ForceRegistry(num_types=4)
        # Disable all but one
        for f in registry.forces:
            f.enabled = False
        registry.forces[0].enabled = True
        active = registry.get_active_forces()
        assert len(active) == 1
    
    def test_pack_for_kernel(self):
        registry = ForceRegistry(num_types=3)
        # Enable just one force
        for f in registry.forces:
            f.enabled = False
        registry.forces[0].enabled = True
        
        matrices, params, extra = registry.pack_for_kernel()
        assert matrices.shape[0] == 1  # One active force
        assert matrices.shape[1] == 3  # 3 types
        assert params.shape == (1, 8)
    
    def test_set_num_types(self):
        registry = ForceRegistry(num_types=4)
        old_shape = registry.forces[0].matrix.shape
        registry.set_num_types(6)
        new_shape = registry.forces[0].matrix.shape
        assert old_shape == (4, 4)
        assert new_shape == (6, 6)
    
    def test_serialization(self):
        registry = ForceRegistry(num_types=3)
        data = registry.to_dict()
        assert 'forces' in data
        assert 'num_types' in data
        
        # Deserialize
        restored = ForceRegistry.from_dict(data)
        assert restored.num_types == registry.num_types
        assert len(restored.forces) == len(registry.forces)
    
    def test_get_max_radius(self):
        registry = ForceRegistry(num_types=4)
        # Enable force with known max radius
        for f in registry.forces:
            f.enabled = False
        registry.forces[0].enabled = True
        registry.forces[0].max_radius = 0.25
        
        assert registry.get_max_radius() == 0.25


class TestPresetRegistries:
    """Test preset force configurations."""
    
    def test_particle_life_preset(self):
        registry = create_preset_registry('particle_life', num_types=5)
        assert len(registry.forces) >= 1
        assert registry.forces[0].name == "Particle Life"
    
    def test_molecular_preset(self):
        registry = create_preset_registry('molecular', num_types=4)
        assert any(f.dropoff == DropoffType.LENNARD_JONES for f in registry.forces)
    
    def test_plasma_preset(self):
        registry = create_preset_registry('plasma', num_types=4)
        assert any(f.dropoff == DropoffType.YUKAWA for f in registry.forces)
    
    def test_gravitational_preset(self):
        registry = create_preset_registry('gravitational', num_types=3)
        assert any(f.dropoff == DropoffType.INVERSE_SQUARE for f in registry.forces)
    
    def test_crystal_preset(self):
        registry = create_preset_registry('crystal', num_types=4)
        assert len(registry.forces) >= 2  # Multiple forces for crystal


class TestForceConservation:
    """Test physical properties of forces."""
    
    def test_newton_third_law(self):
        """Force on A due to B should equal negative force on B due to A."""
        from pyparticles.physics.kernels import compute_forces_multi
        
        # Two particles
        pos = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
        colors = np.array([0, 0], dtype=np.int32)
        angle = np.zeros(2, dtype=np.float32)
        
        # Setup force rules
        matrices = np.array([[[1.0]]], dtype=np.float32)
        params = np.zeros((1, 8), dtype=np.float32)
        params[0] = [0.02, 0.1, 1.0, 0.01, 0, 0, 0, 0]  # Linear force
        
        species = np.array([[0.05, 1.0, 0.0]], dtype=np.float32)
        
        # Grid setup
        grid_counts = np.zeros((4, 4), dtype=np.int32)
        grid_cells = np.zeros((4, 4, 10), dtype=np.int32)
        grid_counts[2, 2] = 2
        grid_cells[2, 2, 0] = 0
        grid_cells[2, 2, 1] = 1
        
        forces, _ = compute_forces_multi(
            pos, colors, angle, 2,
            matrices, params, species,
            0.0, 10.0,  # wave params disabled
            grid_counts, grid_cells, 0.5,
            0.0  # no gravity
        )
        
        # Forces should be equal and opposite
        assert np.allclose(forces[0], -forces[1], atol=1e-6)
