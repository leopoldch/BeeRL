import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.swarm_env import SwarmEnv


class TestSwarmEnvInitialization:
    
    def test_given_default_parameters_when_initialized_then_n_agents_is_set(self):
        expected_n_agents = 4
        env = SwarmEnv(n_agents=expected_n_agents)
        assert env.n_agents == expected_n_agents
    
    def test_given_world_size_when_initialized_then_world_size_is_set(self):
        expected_world_size = 30.0
        env = SwarmEnv(world_size=expected_world_size)
        assert env.world_size == expected_world_size
    
    def test_given_dt_when_initialized_then_dt_is_set(self):
        expected_dt = 0.2
        env = SwarmEnv(dt=expected_dt)
        assert env.dt == expected_dt
    
    def test_given_max_acceleration_when_initialized_then_max_acc_is_set(self):
        expected_max_acc = 2.0
        env = SwarmEnv(max_acc=expected_max_acc)
        assert env.max_acc == expected_max_acc
    
    def test_given_parameters_when_initialized_then_observation_space_has_correct_shape(self):
        n_agents = 5
        expected_obs_dim = n_agents * 4
        env = SwarmEnv(n_agents=n_agents)
        assert env.observation_space.shape == (expected_obs_dim,)
    
    def test_given_parameters_when_initialized_then_action_space_has_correct_shape(self):
        n_agents = 5
        expected_act_dim = n_agents * 2
        env = SwarmEnv(n_agents=n_agents)
        assert env.action_space.shape == (expected_act_dim,)
    
    def test_given_max_acceleration_when_initialized_then_action_space_bounds_are_correct(self):
        max_acc = 2.5
        env = SwarmEnv(max_acc=max_acc)
        assert np.all(env.action_space.high == max_acc)
        assert np.all(env.action_space.low == -max_acc)
    
    def test_given_initialization_when_created_then_swarm_is_initialized(self):
        env = SwarmEnv()
        assert env.swarm is not None
    
    def test_given_initialization_when_created_then_coverage_grid_is_initialized(self):
        env = SwarmEnv()
        assert env.coverage_grid is not None


class TestSwarmEnvReset:
    
    def test_given_env_when_reset_then_returns_observation(self):
        env = SwarmEnv(n_agents=3)
        obs = env.reset()
        assert obs is not None
    
    def test_given_env_when_reset_then_observation_has_correct_shape(self):
        n_agents = 3
        env = SwarmEnv(n_agents=n_agents)
        expected_shape = (n_agents * 4,)
        obs = env.reset()
        assert obs.shape == expected_shape
    
    def test_given_env_when_reset_then_step_counter_is_zero(self):
        env = SwarmEnv()
        env.steps = 100
        env.reset()
        assert env.steps == 0
    
    def test_given_env_with_coverage_when_reset_then_coverage_is_cleared(self):
        env = SwarmEnv()
        env.coverage_grid.grid[:] = True
        env.reset()
        assert env.coverage_grid.get_coverage_percentage() == 0.0
    
    def test_given_env_when_reset_then_observation_is_float32(self):
        env = SwarmEnv()
        obs = env.reset()
        assert obs.dtype == np.float32


class TestSwarmEnvStep:
    
    def test_given_action_when_step_called_then_returns_four_values(self):
        env = SwarmEnv(n_agents=2)
        env.reset()
        action = np.zeros(4)
        result = env.step(action)
        assert len(result) == 4
    
    def test_given_action_when_step_called_then_observation_has_correct_shape(self):
        n_agents = 3
        env = SwarmEnv(n_agents=n_agents)
        env.reset()
        action = np.zeros(n_agents * 2)
        obs, _, _, _ = env.step(action)
        assert obs.shape == (n_agents * 4,)
    
    def test_given_action_when_step_called_then_reward_is_float(self):
        env = SwarmEnv()
        env.reset()
        action = np.zeros(env.action_space.shape)
        _, reward, _, _ = env.step(action)
        assert isinstance(reward, (float, np.floating))
    
    def test_given_action_when_step_called_then_done_is_boolean(self):
        env = SwarmEnv()
        env.reset()
        action = np.zeros(env.action_space.shape)
        _, _, done, _ = env.step(action)
        assert isinstance(done, (bool, np.bool_))
    
    def test_given_action_when_step_called_then_info_contains_coverage(self):
        env = SwarmEnv()
        env.reset()
        action = np.zeros(env.action_space.shape)
        _, _, _, info = env.step(action)
        assert "coverage" in info
    
    def test_given_action_when_step_called_then_info_contains_collisions(self):
        env = SwarmEnv()
        env.reset()
        action = np.zeros(env.action_space.shape)
        _, _, _, info = env.step(action)
        assert "collisions" in info
    
    def test_given_step_when_executed_then_step_counter_increments(self):
        env = SwarmEnv()
        env.reset()
        action = np.zeros(env.action_space.shape)
        initial_steps = env.steps
        env.step(action)
        assert env.steps == initial_steps + 1
    
    def test_given_max_steps_reached_when_step_called_then_done_is_true(self):
        env = SwarmEnv()
        env.reset()
        env.steps = 799
        action = np.zeros(env.action_space.shape)
        _, _, done, _ = env.step(action)
        assert done is True
    
    def test_given_steps_below_max_when_step_called_then_done_is_false(self):
        env = SwarmEnv()
        env.reset()
        action = np.zeros(env.action_space.shape)
        _, _, done, _ = env.step(action)
        assert done is False


class TestSwarmEnvRewardComputation:
    
    @patch.object(SwarmEnv, '_compute_reward')
    def test_given_step_when_executed_then_reward_computation_is_called(self, mock_compute_reward):
        env = SwarmEnv()
        env.reset()
        mock_compute_reward.return_value = 0.0
        action = np.zeros(env.action_space.shape)
        env.step(action)
        mock_compute_reward.assert_called_once()
    
    def test_given_newly_covered_cells_when_reward_computed_then_reward_is_positive(self):
        env = SwarmEnv()
        newly_covered = 5
        collisions = 0
        action = np.zeros(env.action_space.shape)
        reward = env._compute_reward(newly_covered, collisions, action)
        assert reward > 0
    
    def test_given_collisions_when_reward_computed_then_reward_includes_penalty(self):
        env = SwarmEnv()
        newly_covered = 0
        collisions = 2
        action = np.zeros(env.action_space.shape)
        reward = env._compute_reward(newly_covered, collisions, action)
        assert reward < 0
    
    def test_given_large_action_when_reward_computed_then_includes_effort_penalty(self):
        env = SwarmEnv()
        newly_covered = 0
        collisions = 0
        action = np.ones(4) * 5.0
        reward = env._compute_reward(newly_covered, collisions, action)
        assert reward < 0
    
    def test_given_zero_action_when_reward_computed_then_no_effort_penalty(self):
        env = SwarmEnv()
        newly_covered = 1
        collisions = 0
        action = np.zeros(4)
        expected_reward = 1.0
        reward = env._compute_reward(newly_covered, collisions, action)
        assert np.isclose(reward, expected_reward)


class TestSwarmEnvCoverageUpdate:
    
    @patch.object(SwarmEnv, '_update_grid_coverage')
    def test_given_step_when_executed_then_coverage_update_is_called(self, mock_update_coverage):
        env = SwarmEnv()
        env.reset()
        mock_update_coverage.return_value = 0
        action = np.zeros(env.action_space.shape)
        env.step(action)
        mock_update_coverage.assert_called_once()
    
    def test_given_agent_positions_when_coverage_updated_then_returns_newly_covered_count(self):
        env = SwarmEnv(n_agents=2)
        env.reset()
        newly_covered = env._update_grid_coverage()
        assert isinstance(newly_covered, (int, np.integer))
    
    def test_given_agents_in_same_cell_when_coverage_updated_then_counts_cell_once(self):
        env = SwarmEnv(n_agents=2)
        env.reset()
        env.swarm.units[0].pos = np.array([0.0, 0.0])
        env.swarm.units[1].pos = np.array([0.0, 0.0])
        newly_covered = env._update_grid_coverage()
        assert newly_covered == 1


class TestSwarmEnvCollisionDetection:
    
    def test_given_agents_far_apart_when_step_executed_then_no_collisions_in_info(self):
        env = SwarmEnv(n_agents=2)
        env.reset()
        env.swarm.units[0].pos = np.array([0.0, 0.0])
        env.swarm.units[1].pos = np.array([10.0, 10.0])
        action = np.zeros(env.action_space.shape)
        _, _, _, info = env.step(action)
        assert info["collisions"] == 0
    
    def test_given_agents_colliding_when_step_executed_then_collision_count_in_info(self):
        env = SwarmEnv(n_agents=2)
        env.reset()
        env.swarm.units[0].pos = np.array([0.0, 0.0])
        env.swarm.units[1].pos = np.array([0.1, 0.0])
        action = np.zeros(env.action_space.shape)
        _, _, _, info = env.step(action)
        assert info["collisions"] > 0


class TestSwarmEnvObservation:
    
    def test_given_env_when_observation_requested_then_returns_float32_array(self):
        env = SwarmEnv()
        env.reset()
        obs = env._get_obs()
        assert obs.dtype == np.float32
    
    def test_given_env_with_n_agents_when_observation_requested_then_has_correct_size(self):
        n_agents = 6
        env = SwarmEnv(n_agents=n_agents)
        env.reset()
        expected_size = n_agents * 4
        obs = env._get_obs()
        assert len(obs) == expected_size


class TestSwarmEnvRenderAndClose:
    
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.pause')
    def test_given_env_when_render_called_then_matplotlib_is_used(self, mock_pause, mock_scatter, mock_clf):
        env = SwarmEnv()
        env.reset()
        env.render()
        mock_clf.assert_called_once()
    
    def test_given_env_when_close_called_then_no_exception_raised(self):
        env = SwarmEnv()
        try:
            env.close()
            success = True
        except Exception:
            success = False
        assert success is True


class TestSwarmEnvIntegration:
    
    def test_given_full_episode_when_executed_then_completes_successfully(self):
        env = SwarmEnv(n_agents=2)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            steps += 1
        assert steps > 0
    
    def test_given_episode_when_executed_then_coverage_increases_over_time(self):
        env = SwarmEnv(n_agents=4)
        env.reset()
        initial_coverage = env.coverage_grid.get_coverage_percentage()
        for _ in range(50):
            action = env.action_space.sample()
            env.step(action)
        final_coverage = env.coverage_grid.get_coverage_percentage()
        assert final_coverage >= initial_coverage