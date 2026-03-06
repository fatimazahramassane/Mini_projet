

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  

def generate_grid(difficulty='easy'):
    """
    Generate a 2D grid with obstacles, start, and goal based on difficulty level.
    - easy: small grid, few obstacles.
    - medium: medium grid, more obstacles.
    - hard: large grid, complex obstacles.
    
    Returns:
        grid (np.array): 2D array where 0=free, 1=obstacle.
        start (tuple): (row, col) start position.
        goal (tuple): (row, col) goal position.
    """
    if difficulty == 'easy':
        size = 5  # 5x5 grid
        grid = np.zeros((size, size), dtype=int)
        # Place some obstacles
        obstacles = [(1,1), (1,2), (2,3)]
        for obs in obstacles:
            grid[obs] = 1
        start = (0, 0)
        goal = (4, 4)
    elif difficulty == 'medium':
        size = 10  # 10x10 grid
        grid = np.zeros((size, size), dtype=int)
        # More obstacles, like walls
        for i in range(3, 7):
            grid[i, 2] = 1  # Vertical wall
            grid[2, i] = 1  # Horizontal wall
        grid[5,5] = 1  # Isolated obstacle
        start = (0, 0)
        goal = (9, 9)
    elif difficulty == 'hard':
        size = 15  # 15x15 grid
        grid = np.zeros((size, size), dtype=int)
        # Complex obstacles: multiple walls and clusters
        for i in range(4, 10):
            grid[i, 3] = 1  # Wall 1
            grid[5, i] = 1  # Wall 2
        for i in range(7, 12):
            grid[i, 8] = 1  # Wall 3
            grid[10, i] = 1  # Wall 4
        # Random obstacles
        num_random = 10
        for _ in range(num_random):
            r, c = np.random.randint(0, size, 2)
            if (r, c) not in [(0,0), (14,14)]:  # Avoid start/goal
                grid[r, c] = 1
        start = (0, 0)
        goal = (14, 14)
    else:
        raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'.")
    
    # Ensure start and goal are free
    grid[start] = 0
    grid[goal] = 0
    
    return grid, start, goal

def is_valid_position(grid, pos):
    """Check if position is within bounds and not an obstacle."""
    rows, cols = grid.shape
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0

def get_neighbors(grid, pos):
    """Get valid 4-neighbors for a position."""
    neighbors = []
    for dr, dc in DIRECTIONS:
        new_pos = (pos[0] + dr, pos[1] + dc)
        if is_valid_position(grid, new_pos):
            neighbors.append(new_pos)
    return neighbors

def plot_grid(grid, start, goal, path=None, filename='grid.png'):

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='binary', origin='upper')  # Black for obstacles (1), white for free (0)
    
    # Mark start and goal
    ax.plot(start[1], start[0], 'go', markersize=10, label='Start')  # Green circle
    ax.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')    # Red circle
    
    if path:
        path_r = [p[0] for p in path]
        path_c = [p[1] for p in path]
        ax.plot(path_c, path_r, 'b-', linewidth=2, label='Path')  # Blue line for path
    
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
    ax.grid(True, color='gray', linewidth=0.5)
    ax.set_title('Grid Environment')
    ax.legend()
    
    # Save to file
    os.makedirs('figures', exist_ok=True)
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Grid plot saved to {filepath}")


if __name__ == "__main__":
    
    grid_easy, start_easy, goal_easy = generate_grid('easy')
    plot_grid(grid_easy, start_easy, goal_easy, filename='easy_grid.png')

    grid_easy, start_easy, goal_easy = generate_grid('medium')
    plot_grid(grid_easy, start_easy, goal_easy, filename='medium.png')

    
    grid_easy, start_easy, goal_easy = generate_grid('hard')
    plot_grid(grid_easy, start_easy, goal_easy, filename='hard.png')