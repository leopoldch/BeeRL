import pytest
import numpy as np
from src.coverage_grid import CoverageGrid


class TestCoverageGridInitialization:
    
    def test_given_default_parameters_when_initialized_then_world_size_is_set(self):
        expected_world_size = 20.0
        grid = CoverageGrid(world_size=expected_world_size)
        assert grid.world_size == expected_world_size
    
    def test_given_grid_resolution_when_initialized_then_resolution_is_set(self):
        expected_resolution = 40
        grid = CoverageGrid(grid_res=expected_resolution)
        assert grid.grid_res == expected_resolution
    
    def test_given_parameters_when_initialized_then_cell_size_calculated_correctly(self):
        world_size = 20.0
        grid_res = 40
        expected_cell_size = world_size / grid_res
        grid = CoverageGrid(world_size=world_size, grid_res=grid_res)
        assert grid.cell_size == expected_cell_size
    
    def test_given_initialization_when_grid_created_then_all_cells_are_unvisited(self):
        grid = CoverageGrid(grid_res=10)
        assert np.all(grid.grid == False)
    
    def test_given_grid_resolution_when_initialized_then_grid_has_correct_shape(self):
        grid_res = 50
        grid = CoverageGrid(grid_res=grid_res)
        assert grid.grid.shape == (grid_res, grid_res)


class TestCoverageGridReset:
    
    def test_given_empty_grid_when_reset_then_grid_remains_empty(self):
        grid = CoverageGrid(grid_res=10)
        grid.reset()
        assert np.all(grid.grid == False)
    
    def test_given_marked_grid_when_reset_then_all_cells_become_unvisited(self):
        grid = CoverageGrid(grid_res=10)
        grid.grid[5, 5] = True
        grid.grid[3, 7] = True
        grid.reset()
        assert np.all(grid.grid == False)
    
    def test_given_fully_covered_grid_when_reset_then_coverage_becomes_zero(self):
        grid = CoverageGrid(grid_res=10)
        grid.grid[:] = True
        grid.reset()
        assert grid.get_coverage_percentage() == 0.0


class TestCoverageGridCoordinateConversion:
    
    def test_given_center_position_when_converted_then_returns_center_grid_coords(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        center_pos = np.array([0.0, 0.0])
        expected_gx = 20
        expected_gy = 20
        gx, gy = grid.world_to_grid_coords(center_pos)
        assert gx == expected_gx and gy == expected_gy
    
    def test_given_negative_position_when_converted_then_returns_low_grid_coords(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([-5.0, -5.0])
        gx, gy = grid.world_to_grid_coords(pos)
        assert gx < 20 and gy < 20
    
    def test_given_positive_position_when_converted_then_returns_high_grid_coords(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([5.0, 5.0])
        gx, gy = grid.world_to_grid_coords(pos)
        assert gx > 20 and gy > 20
    
    def test_given_out_of_bounds_position_when_converted_then_coords_clamped_to_valid_range(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([100.0, 100.0])
        gx, gy = grid.world_to_grid_coords(pos)
        assert 0 <= gx < 40 and 0 <= gy < 40
    
    def test_given_negative_out_of_bounds_when_converted_then_coords_clamped_to_zero(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([-100.0, -100.0])
        gx, gy = grid.world_to_grid_coords(pos)
        assert gx == 0 and gy == 0


class TestCoverageGridMarkPosition:
    
    def test_given_unvisited_cell_when_marked_then_returns_true(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([0.0, 0.0])
        is_newly_covered = grid.mark_position(pos)
        assert is_newly_covered == True
    
    def test_given_already_visited_cell_when_marked_then_returns_false(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([0.0, 0.0])
        grid.mark_position(pos)
        is_newly_covered = grid.mark_position(pos)
        assert is_newly_covered == False
    
    def test_given_position_when_marked_then_cell_becomes_visited(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([0.0, 0.0])
        grid.mark_position(pos)
        assert grid.is_visited(pos) == True
    
    def test_given_multiple_positions_when_marked_then_correct_count_returned(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        newly_covered = grid.mark_positions(positions)
        assert newly_covered == 3
    
    def test_given_duplicate_positions_when_marked_then_counts_only_new_cells(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        positions = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        newly_covered = grid.mark_positions(positions)
        assert newly_covered == 2


class TestCoverageGridVisitedCheck:
    
    def test_given_unvisited_position_when_checked_then_returns_false(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([0.0, 0.0])
        is_visited = grid.is_visited(pos)
        assert is_visited == False
    
    def test_given_visited_position_when_checked_then_returns_true(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        pos = np.array([0.0, 0.0])
        grid.mark_position(pos)
        is_visited = grid.is_visited(pos)
        assert is_visited == True
    
    def test_given_nearby_visited_position_when_different_cell_checked_then_returns_false(self):
        grid = CoverageGrid(world_size=20.0, grid_res=40)
        grid.mark_position(np.array([0.0, 0.0]))
        different_cell_pos = np.array([5.0, 5.0])
        is_visited = grid.is_visited(different_cell_pos)
        assert is_visited == False


class TestCoverageGridStatistics:
    
    def test_given_empty_grid_when_coverage_percentage_requested_then_returns_zero(self):
        grid = CoverageGrid(grid_res=10)
        coverage = grid.get_coverage_percentage()
        assert coverage == 0.0
    
    def test_given_fully_covered_grid_when_coverage_percentage_requested_then_returns_one(self):
        grid = CoverageGrid(grid_res=10)
        grid.grid[:] = True
        coverage = grid.get_coverage_percentage()
        assert coverage == 1.0
    
    def test_given_half_covered_grid_when_coverage_percentage_requested_then_returns_half(self):
        grid = CoverageGrid(grid_res=10)
        grid.grid[:5, :] = True
        coverage = grid.get_coverage_percentage()
        assert coverage == 0.5
    
    def test_given_empty_grid_when_coverage_count_requested_then_returns_zero(self):
        grid = CoverageGrid(grid_res=10)
        count = grid.get_coverage_count()
        assert count == 0
    
    def test_given_marked_cells_when_coverage_count_requested_then_returns_correct_count(self):
        grid = CoverageGrid(grid_res=10)
        grid.grid[0, 0] = True
        grid.grid[1, 1] = True
        grid.grid[2, 2] = True
        count = grid.get_coverage_count()
        assert count == 3
    
    def test_given_grid_when_total_cells_requested_then_returns_correct_total(self):
        grid_res = 10
        grid = CoverageGrid(grid_res=grid_res)
        expected_total = grid_res * grid_res
        total = grid.get_total_cells()
        assert total == expected_total


class TestCoverageGridGetters:
    
    def test_given_grid_when_get_grid_called_then_returns_numpy_array(self):
        grid = CoverageGrid(grid_res=10)
        grid_array = grid.get_grid()
        assert isinstance(grid_array, np.ndarray)
    
    def test_given_grid_when_get_grid_called_then_returns_correct_shape(self):
        grid_res = 15
        grid = CoverageGrid(grid_res=grid_res)
        grid_array = grid.get_grid()
        assert grid_array.shape == (grid_res, grid_res)
    
    def test_given_empty_grid_when_uncovered_positions_requested_then_returns_all_positions(self):
        grid_res = 5
        grid = CoverageGrid(world_size=10.0, grid_res=grid_res)
        expected_count = grid_res * grid_res
        uncovered = grid.get_uncovered_positions()
        assert len(uncovered) == expected_count
    
    def test_given_fully_covered_grid_when_uncovered_positions_requested_then_returns_empty_array(self):
        grid = CoverageGrid(grid_res=5)
        grid.grid[:] = True
        uncovered = grid.get_uncovered_positions()
        assert len(uncovered) == 0
    
    def test_given_partially_covered_when_uncovered_positions_requested_then_returns_correct_count(self):
        grid = CoverageGrid(grid_res=10)
        grid.grid[0, 0] = True
        grid.grid[1, 1] = True
        expected_uncovered = 10 * 10 - 2
        uncovered = grid.get_uncovered_positions()
        assert len(uncovered) == expected_uncovered
    
    def test_given_grid_when_uncovered_positions_requested_then_returns_world_coordinates(self):
        world_size = 20.0
        grid = CoverageGrid(world_size=world_size, grid_res=10)
        half_world = world_size / 2.0
        uncovered = grid.get_uncovered_positions()
        assert np.all(uncovered >= -half_world) and np.all(uncovered <= half_world)
    
    def test_given_uncovered_positions_when_requested_then_array_has_correct_shape(self):
        grid = CoverageGrid(grid_res=5)
        uncovered = grid.get_uncovered_positions()
        assert uncovered.shape[1] == 2