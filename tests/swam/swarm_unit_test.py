import pytest
import numpy as np
from src.swarm.swarm_unit import SwarmUnit


class TestSwarmUnitInitialization:
    
    def test_given_no_parameters_when_initialized_then_position_is_zero_vector(self):
        unit = SwarmUnit()
        assert np.allclose(unit.pos, np.zeros(2))
    
    def test_given_no_parameters_when_initialized_then_velocity_is_zero_vector(self):
        unit = SwarmUnit()
        assert np.allclose(unit.vel, np.zeros(2))
    
    def test_given_custom_position_when_initialized_then_position_is_set_correctly(self):
        expected_pos = np.array([1.5, 2.5])
        unit = SwarmUnit(pos=expected_pos)
        assert np.allclose(unit.pos, expected_pos)
    
    def test_given_custom_velocity_when_initialized_then_velocity_is_set_correctly(self):
        expected_vel = np.array([0.5, -0.3])
        unit = SwarmUnit(vel=expected_vel)
        assert np.allclose(unit.vel, expected_vel)
    
    def test_given_unit_id_when_initialized_then_id_is_set_correctly(self):
        expected_id = 42
        unit = SwarmUnit(unit_id=expected_id)
        assert unit.id == expected_id


class TestSwarmUnitAcceleration:
    
    def test_given_zero_velocity_when_acceleration_applied_then_velocity_increases(self):
        unit = SwarmUnit()
        acceleration = np.array([1.0, 0.0])
        dt = 0.1
        expected_vel = np.array([0.1, 0.0])
        unit.apply_acceleration(acceleration, dt)
        assert np.allclose(unit.vel, expected_vel)
    
    def test_given_existing_velocity_when_acceleration_applied_then_velocity_accumulates(self):
        initial_vel = np.array([0.5, 0.3])
        unit = SwarmUnit(vel=initial_vel.copy())
        acceleration = np.array([1.0, -0.5])
        dt = 0.1
        expected_vel = initial_vel + acceleration * dt
        unit.apply_acceleration(acceleration, dt)
        assert np.allclose(unit.vel, expected_vel)
    
    def test_given_negative_acceleration_when_applied_then_velocity_decreases(self):
        unit = SwarmUnit(vel=np.array([1.0, 1.0]))
        acceleration = np.array([-2.0, -2.0])
        dt = 0.1
        expected_vel = np.array([0.8, 0.8])
        unit.apply_acceleration(acceleration, dt)
        assert np.allclose(unit.vel, expected_vel)


class TestSwarmUnitPosition:
    
    def test_given_zero_velocity_when_position_updated_then_position_unchanged(self):
        initial_pos = np.array([1.0, 2.0])
        unit = SwarmUnit(pos=initial_pos.copy())
        dt = 0.1
        unit.update_position(dt)
        assert np.allclose(unit.pos, initial_pos)
    
    def test_given_positive_velocity_when_position_updated_then_position_increases(self):
        initial_pos = np.array([1.0, 2.0])
        velocity = np.array([0.5, 0.3])
        unit = SwarmUnit(pos=initial_pos.copy(), vel=velocity)
        dt = 0.1
        expected_pos = initial_pos + velocity * dt
        unit.update_position(dt)
        assert np.allclose(unit.pos, expected_pos)
    
    def test_given_negative_velocity_when_position_updated_then_position_decreases(self):
        initial_pos = np.array([1.0, 2.0])
        velocity = np.array([-1.0, -0.5])
        unit = SwarmUnit(pos=initial_pos.copy(), vel=velocity)
        dt = 0.1
        expected_pos = initial_pos + velocity * dt
        unit.update_position(dt)
        assert np.allclose(unit.pos, expected_pos)


class TestSwarmUnitBoundaryClipping:
    
    def test_given_position_within_bounds_when_clipped_then_position_unchanged(self):
        position = np.array([0.0, 0.0])
        unit = SwarmUnit(pos=position.copy())
        min_bound = -5.0
        max_bound = 5.0
        unit.clip_position(min_bound, max_bound)
        assert np.allclose(unit.pos, position)
    
    def test_given_position_exceeds_max_bound_when_clipped_then_position_clamped_to_max(self):
        unit = SwarmUnit(pos=np.array([10.0, 8.0]))
        min_bound = -5.0
        max_bound = 5.0
        expected_pos = np.array([5.0, 5.0])
        unit.clip_position(min_bound, max_bound)
        assert np.allclose(unit.pos, expected_pos)
    
    def test_given_position_below_min_bound_when_clipped_then_position_clamped_to_min(self):
        unit = SwarmUnit(pos=np.array([-10.0, -8.0]))
        min_bound = -5.0
        max_bound = 5.0
        expected_pos = np.array([-5.0, -5.0])
        unit.clip_position(min_bound, max_bound)
        assert np.allclose(unit.pos, expected_pos)


class TestSwarmUnitDistance:
    
    def test_given_same_position_when_distance_calculated_then_distance_is_zero(self):
        position = np.array([1.0, 2.0])
        unit1 = SwarmUnit(pos=position.copy())
        unit2 = SwarmUnit(pos=position.copy())
        distance = unit1.distance_to(unit2)
        assert np.isclose(distance, 0.0)
    
    def test_given_horizontal_separation_when_distance_calculated_then_distance_is_correct(self):
        unit1 = SwarmUnit(pos=np.array([0.0, 0.0]))
        unit2 = SwarmUnit(pos=np.array([3.0, 0.0]))
        expected_distance = 3.0
        distance = unit1.distance_to(unit2)
        assert np.isclose(distance, expected_distance)
    
    def test_given_diagonal_separation_when_distance_calculated_then_distance_is_correct(self):
        unit1 = SwarmUnit(pos=np.array([0.0, 0.0]))
        unit2 = SwarmUnit(pos=np.array([3.0, 4.0]))
        expected_distance = 5.0
        distance = unit1.distance_to(unit2)
        assert np.isclose(distance, expected_distance)


class TestSwarmUnitStateManagement:
    
    def test_given_unit_when_get_state_called_then_returns_concatenated_pos_vel(self):
        pos = np.array([1.0, 2.0])
        vel = np.array([0.5, -0.3])
        unit = SwarmUnit(pos=pos, vel=vel)
        expected_state = np.array([1.0, 2.0, 0.5, -0.3])
        state = unit.get_state()
        assert np.allclose(state, expected_state)
    
    def test_given_new_state_when_set_state_called_then_position_is_updated(self):
        unit = SwarmUnit()
        new_pos = np.array([3.0, 4.0])
        new_vel = np.array([1.0, 1.0])
        unit.set_state(new_pos, new_vel)
        assert np.allclose(unit.pos, new_pos)
    
    def test_given_new_state_when_set_state_called_then_velocity_is_updated(self):
        unit = SwarmUnit()
        new_pos = np.array([3.0, 4.0])
        new_vel = np.array([1.0, 1.0])
        unit.set_state(new_pos, new_vel)
        assert np.allclose(unit.vel, new_vel)
    
    def test_given_list_input_when_set_state_called_then_converts_to_numpy_array(self):
        unit = SwarmUnit()
        new_pos = [1.0, 2.0]
        new_vel = [0.5, 0.3]
        unit.set_state(new_pos, new_vel)
        assert isinstance(unit.pos, np.ndarray)