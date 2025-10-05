import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.swarm.swarm import Swarm
from src.swarm.swarm_unit import SwarmUnit


class TestSwarmInitialization:
    
    def test_given_default_parameters_when_initialized_then_creates_correct_number_of_units(self):
        n_agents = 4
        swarm = Swarm(n_agents=n_agents)
        assert len(swarm.units) == n_agents
    
    def test_given_custom_agent_count_when_initialized_then_creates_correct_number_of_units(self):
        n_agents = 10
        swarm = Swarm(n_agents=n_agents)
        assert len(swarm.units) == n_agents
    
    def test_given_world_size_when_initialized_then_world_size_is_set_correctly(self):
        expected_world_size = 50.0
        swarm = Swarm(world_size=expected_world_size)
        assert swarm.world_size == expected_world_size
    
    def test_given_collision_distance_when_initialized_then_collision_dist_is_set_correctly(self):
        expected_collision_dist = 0.5
        swarm = Swarm(collision_dist=expected_collision_dist)
        assert swarm.collision_dist == expected_collision_dist
    
    def test_given_initialization_when_units_created_then_each_unit_has_unique_id(self):
        n_agents = 5
        swarm = Swarm(n_agents=n_agents)
        unit_ids = [unit.id for unit in swarm.units]
        assert len(set(unit_ids)) == n_agents


class TestSwarmReset:
    
    def test_given_swarm_when_reset_then_all_units_have_zero_velocity(self):
        swarm = Swarm(n_agents=4)
        swarm.reset()
        velocities = swarm.get_velocities()
        assert np.allclose(velocities, np.zeros((4, 2)))
    
    def test_given_swarm_when_reset_then_positions_are_within_spawn_area(self):
        world_size = 20.0
        spawn_ratio = 0.8
        max_spawn_coord = world_size * spawn_ratio / 2.0
        swarm = Swarm(n_agents=4, world_size=world_size)
        swarm.reset(spawn_area_ratio=spawn_ratio)
        positions = swarm.get_positions()
        assert np.all(np.abs(positions) <= max_spawn_coord)
    
    def test_given_different_spawn_ratio_when_reset_then_positions_respect_new_ratio(self):
        world_size = 20.0
        spawn_ratio = 0.5
        max_spawn_coord = world_size * spawn_ratio / 2.0
        swarm = Swarm(n_agents=4, world_size=world_size)
        swarm.reset(spawn_area_ratio=spawn_ratio)
        positions = swarm.get_positions()
        assert np.all(np.abs(positions) <= max_spawn_coord)


class TestSwarmActions:
    
    def test_given_zero_actions_when_applied_then_velocities_remain_unchanged(self):
        swarm = Swarm(n_agents=2)
        swarm.reset()
        actions = np.zeros(4)
        dt = 0.1
        max_acc = 1.0
        initial_velocities = swarm.get_velocities().copy()
        swarm.apply_actions(actions, dt, max_acc)
        assert np.allclose(swarm.get_velocities(), initial_velocities)
    
    def test_given_actions_when_applied_then_positions_are_updated(self):
        swarm = Swarm(n_agents=2)
        swarm.reset()
        initial_positions = swarm.get_positions().copy()
        actions = np.array([1.0, 0.0, 0.0, 1.0])
        dt = 0.1
        max_acc = 1.0
        swarm.apply_actions(actions, dt, max_acc)
        assert not np.allclose(swarm.get_positions(), initial_positions)
    
    def test_given_actions_exceeding_max_when_applied_then_actions_are_clipped(self):
        swarm = Swarm(n_agents=1)
        swarm.reset()
        actions = np.array([10.0, 10.0])
        dt = 0.1
        max_acc = 1.0
        swarm.apply_actions(actions, dt, max_acc)
        velocities = swarm.get_velocities()
        expected_max_velocity = max_acc * dt
        assert np.all(np.abs(velocities) <= expected_max_velocity)
    
    def test_given_negative_actions_when_applied_then_velocities_decrease(self):
        swarm = Swarm(n_agents=1)
        swarm.units[0].set_state(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        actions = np.array([-5.0, -5.0])
        dt = 0.1
        max_acc = 1.0
        swarm.apply_actions(actions, dt, max_acc)
        assert np.all(swarm.get_velocities() < 1.0)


class TestSwarmPositionClipping:
    
    def test_given_positions_within_bounds_when_clipped_then_positions_unchanged(self):
        swarm = Swarm(n_agents=2, world_size=20.0)
        swarm.units[0].pos = np.array([0.0, 0.0])
        swarm.units[1].pos = np.array([5.0, -5.0])
        expected_positions = swarm.get_positions().copy()
        swarm.clip_positions()
        assert np.allclose(swarm.get_positions(), expected_positions)
    
    def test_given_positions_exceed_bounds_when_clipped_then_positions_clamped(self):
        world_size = 20.0
        swarm = Swarm(n_agents=1, world_size=world_size)
        swarm.units[0].pos = np.array([50.0, -50.0])
        max_bound = world_size / 2.0
        swarm.clip_positions()
        assert np.all(np.abs(swarm.get_positions()) <= max_bound)


class TestSwarmCollisionDetection:
    
    def test_given_no_units_close_when_collision_detected_then_returns_zero(self):
        swarm = Swarm(n_agents=2, collision_dist=0.1)
        swarm.units[0].pos = np.array([0.0, 0.0])
        swarm.units[1].pos = np.array([10.0, 10.0])
        collisions = swarm.detect_collisions()
        assert collisions == 0
    
    def test_given_two_units_colliding_when_detected_then_returns_one(self):
        swarm = Swarm(n_agents=2, collision_dist=1.0)
        swarm.units[0].pos = np.array([0.0, 0.0])
        swarm.units[1].pos = np.array([0.5, 0.0])
        collisions = swarm.detect_collisions()
        assert collisions == 1
    
    def test_given_three_units_all_colliding_when_detected_then_returns_three(self):
        swarm = Swarm(n_agents=3, collision_dist=1.0)
        swarm.units[0].pos = np.array([0.0, 0.0])
        swarm.units[1].pos = np.array([0.5, 0.0])
        swarm.units[2].pos = np.array([0.25, 0.0])
        collisions = swarm.detect_collisions()
        assert collisions == 3
    
    def test_given_units_at_exact_collision_distance_when_detected_then_no_collision(self):
        collision_dist = 1.0
        swarm = Swarm(n_agents=2, collision_dist=collision_dist)
        swarm.units[0].pos = np.array([0.0, 0.0])
        swarm.units[1].pos = np.array([1.0, 0.0])
        collisions = swarm.detect_collisions()
        assert collisions == 0


class TestSwarmGetters:
    
    def test_given_swarm_when_get_positions_called_then_returns_correct_shape(self):
        n_agents = 5
        swarm = Swarm(n_agents=n_agents)
        positions = swarm.get_positions()
        assert positions.shape == (n_agents, 2)
    
    def test_given_swarm_when_get_velocities_called_then_returns_correct_shape(self):
        n_agents = 5
        swarm = Swarm(n_agents=n_agents)
        velocities = swarm.get_velocities()
        assert velocities.shape == (n_agents, 2)
    
    def test_given_swarm_when_get_state_called_then_returns_flattened_array(self):
        n_agents = 3
        swarm = Swarm(n_agents=n_agents)
        expected_length = n_agents * 4
        state = swarm.get_state()
        assert len(state) == expected_length
    
    def test_given_swarm_when_get_unit_called_then_returns_correct_unit(self):
        swarm = Swarm(n_agents=5)
        target_index = 2
        unit = swarm.get_unit(target_index)
        assert unit.id == target_index
    
    def test_given_swarm_with_set_positions_when_get_positions_called_then_returns_set_values(self):
        swarm = Swarm(n_agents=2)
        swarm.units[0].pos = np.array([1.0, 2.0])
        swarm.units[1].pos = np.array([3.0, 4.0])
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        positions = swarm.get_positions()
        assert np.allclose(positions, expected)