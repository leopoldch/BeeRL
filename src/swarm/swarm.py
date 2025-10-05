import numpy as np
from src.swarm.swarm_unit import SwarmUnit

class Swarm:
    """Manages a group of swarm units"""
    
    def __init__(self, n_agents=4, world_size=20.0, collision_dist=0.2):
        self.n_agents = n_agents
        self.world_size = world_size
        self.collision_dist = collision_dist
        self.units = [SwarmUnit(unit_id=i) for i in range(n_agents)]
        
    def reset(self, spawn_area_ratio=0.8):
        spawn_area = self.world_size * spawn_area_ratio
        for unit in self.units:
            pos = (np.random.rand(2) - 0.5) * spawn_area
            vel = np.zeros(2)
            unit.set_state(pos, vel)
    
    def apply_actions(self, actions, dt, max_acc):
        actions = np.clip(actions.reshape(self.n_agents, 2), -max_acc, max_acc)
        
        for i, unit in enumerate(self.units):
            unit.apply_acceleration(actions[i], dt)
            unit.update_position(dt)
    
    def clip_positions(self):
        half_world = self.world_size / 2.0
        for unit in self.units:
            unit.clip_position(-half_world, half_world)
    
    def detect_collisions(self):
        collisions = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self.units[i].distance_to(self.units[j]) < self.collision_dist:
                    collisions += 1
        return collisions
    
    def get_positions(self):
        return np.array([unit.pos for unit in self.units])
    
    def get_velocities(self):
        return np.array([unit.vel for unit in self.units])
    
    def get_state(self):
        return np.concatenate([unit.get_state() for unit in self.units])
    
    def get_unit(self, index):
        return self.units[index]