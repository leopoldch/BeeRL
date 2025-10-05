import numpy as np

class CoverageGrid:
    
    def __init__(self, world_size=20.0, grid_res=40):
        self.world_size = world_size
        self.grid_res = grid_res
        self.cell_size = world_size / grid_res
        
        self.grid = np.zeros((grid_res, grid_res), dtype=np.bool_)
        
    def reset(self):
        self.grid[:] = False
        
    def world_to_grid_coords(self, world_pos):
        half_world = self.world_size / 2.0
        
        gx = int((world_pos[0] + half_world) / self.cell_size)
        gy = int((world_pos[1] + half_world) / self.cell_size)
        
        gx = np.clip(gx, 0, self.grid_res - 1)
        gy = np.clip(gy, 0, self.grid_res - 1)
        
        return gx, gy
    
    def mark_position(self, world_pos):
        gx, gy = self.world_to_grid_coords(world_pos)
        if not self.grid[gx, gy]:
            self.grid[gx, gy] = True
            return True
        return False
    
    def mark_positions(self, positions):
        newly_covered = 0
        for pos in positions:
            if self.mark_position(pos):
                newly_covered += 1
        return newly_covered
    
    def is_visited(self, world_pos):
        gx, gy = self.world_to_grid_coords(world_pos)
        return self.grid[gx, gy]
    
    def get_coverage_percentage(self):
        return np.mean(self.grid)
    
    def get_coverage_count(self):
        return np.sum(self.grid)
    
    def get_total_cells(self):
        return self.grid_res * self.grid_res
    
    def get_grid(self):
        return self.grid
    
    def get_uncovered_positions(self):
        uncovered_indices = np.argwhere(~self.grid)
        half_world = self.world_size / 2.0

        positions = []
        for gx, gy in uncovered_indices:
            x = (gx + 0.5) * self.cell_size - half_world
            y = (gy + 0.5) * self.cell_size - half_world
            positions.append([x, y])
        
        return np.array(positions) if positions else np.empty((0, 2))