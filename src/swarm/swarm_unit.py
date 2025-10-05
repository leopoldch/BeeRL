import numpy as np

class SwarmUnit:
    
    def __init__(self, pos=None, vel=None, unit_id=0):
        self.id = unit_id
        self.pos = pos if pos is not None else np.zeros(2)
        self.vel = vel if vel is not None else np.zeros(2)
        
    def apply_acceleration(self, acceleration, dt):
        self.vel += acceleration * dt
        
    def update_position(self, dt):
        self.pos += self.vel * dt
        
    def clip_position(self, min_bound, max_bound):
        self.pos = np.clip(self.pos, min_bound, max_bound)
        
    def distance_to(self, other_unit):
        return np.linalg.norm(self.pos - other_unit.pos)
    
    def get_state(self):
        return np.concatenate([self.pos, self.vel])
    
    def set_state(self, pos, vel):
        self.pos = np.array(pos)
        self.vel = np.array(vel)