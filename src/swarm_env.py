import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from src.swarm.swarm import Swarm
from src.coverage_grid import CoverageGrid

class SwarmEnv(gym.Env):
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, n_agents=4, world_size=20.0, dt=0.1, max_acc=1.0, grid_res=40):
        super().__init__()
        
        self.n_agents = n_agents
        self.world_size = world_size
        self.dt = dt
        self.max_acc = max_acc
        self.grid_res = grid_res
        
        self.swarm = Swarm(n_agents=n_agents, world_size=world_size, collision_dist=0.2)
        
        self.coverage_grid = CoverageGrid(world_size=world_size, grid_res=grid_res)
        
        obs_dim = self.n_agents * 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        act_dim = self.n_agents * 2
        self.action_space = spaces.Box(
            low=-self.max_acc, high=self.max_acc, shape=(act_dim,), dtype=np.float32
        )
        
        self.steps = 0
        self.reset()
    
    def reset(self):
        self.swarm.reset()
        self.coverage_grid.reset()
        self.steps = 0
        return self._get_obs()
    
    def step(self, action):
        self.swarm.apply_actions(action, self.dt, self.max_acc)
        self.swarm.clip_positions()
        
        newly_covered = self._update_grid_coverage()
        collisions = self.swarm.detect_collisions()
        
        reward = self._compute_reward(newly_covered, collisions, action)
        
        self.steps += 1
        done = (self.steps >= 800)
        info = {
            "coverage": self.coverage_grid.get_coverage_percentage(),
            "collisions": collisions,
            "covered_cells": self.coverage_grid.get_coverage_count()
        }
        
        return self._get_obs(), reward, done, info
    
    def _update_grid_coverage(self):
        positions = self.swarm.get_positions()
        return self.coverage_grid.mark_positions(positions)
    
    def _compute_reward(self, newly_covered, collisions, action):
        collision_penalty = -10.0 * collisions
        coverage_reward = float(newly_covered)
        effort_penalty = -0.01 * np.sum(action**2)
        return coverage_reward + collision_penalty + effort_penalty
    
    def _get_obs(self):
        return self.swarm.get_state().astype(np.float32)
    
    def render(self, mode='human'):
        plt.clf()
        half_world = self.world_size / 2
        
        plt.xlim(-half_world, half_world)
        plt.ylim(-half_world, half_world)
        
        positions = self.swarm.get_positions()
        plt.scatter(positions[:, 0], positions[:, 1], s=80)
        plt.gca().set_aspect('equal', 'box')
        plt.pause(0.001)
    
    def close(self):
        pass