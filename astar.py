

import heapq
import time
from collections import namedtuple
import numpy as np


np.random.seed(42)


Node = namedtuple('Node', ['pos', 'g', 'h', 'f', 'parent'])

def manhattan_heuristic(pos, goal):
    """
    Manhattan heuristic: h(n) = |x - xg| + |y - yg|
    Admissible and consistent for 4-neighbors grid with unit costs.
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def zero_heuristic(pos, goal):
    """Zero heuristic: h(n)=0, equivalent to UCS/Dijkstra."""
    return 0

def search(grid, start, goal, heuristic=manhattan_heuristic, weight=1.0):
    """
    Generalized search function.
    - If weight=1 and heuristic != zero: A*
    - If heuristic=zero: UCS (f=g)
    - If g ignored (f=h): Greedy
    - If weight>1: Weighted A*
    
    Returns:
        path (list): List of positions from start to goal.
        cost (float): Total cost of the path.
        expanded (int): Number of nodes expanded.
        max_open_size (int): Maximum size of OPEN.
        exec_time (float): Execution time in seconds.
    """
    # To handle Greedy separately (f=h, ignore g)
    is_greedy = (heuristic != zero_heuristic and weight == 0)  # weight=0 to flag Greedy
    
    open_set = []  # Priority queue (heap) for OPEN
    heapq.heappush(open_set, (0, 0, start, None))  # (f, tie-breaker, pos, parent)
    came_from = {}  # parent pointers
    g_score = {start: 0}  # g(n)
    closed = set()  # CLOSED set
    expanded = 0  # Nodes expanded (popped from OPEN)
    max_open_size = 0  # Max size of OPEN
    tie_breaker = 0  # To avoid heap comparisons issues
    
    start_time = time.time()
    
    while open_set:
        max_open_size = max(max_open_size, len(open_set))
        
        current_f, _, current, parent = heapq.heappop(open_set)
        expanded += 1
        
        if current in closed:
            continue
        closed.add(current)
        
        # Reconstruct if goal reached
        if current == goal:
            path = []
            cost = g_score[current]
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            path.reverse()
            exec_time = time.time() - start_time
            return path, cost, expanded, max_open_size, exec_time
        
        # Get neighbors
        from grid import get_neighbors  
        neighbors = get_neighbors(grid, current)
        
        for neighbor in neighbors:
            if neighbor in closed:
                continue
            
            # Tentative g
            tentative_g = g_score[current] + 1  
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h = heuristic(neighbor, goal)
                if is_greedy:
                    f = h  # Greedy: f = h
                else:
                    f = tentative_g + weight * h  # A* or Weighted
                came_from[neighbor] = current
                heapq.heappush(open_set, (f, tie_breaker, neighbor, current))
                tie_breaker += 1
    
    # No path found
    exec_time = time.time() - start_time
    return None, np.inf, expanded, max_open_size, exec_time

def run_astar(grid, start, goal):
    """Run standard A* with Manhattan."""
    return search(grid, start, goal, heuristic=manhattan_heuristic, weight=1.0)

def run_ucs(grid, start, goal):
    """Run UCS (A* with h=0)."""
    return search(grid, start, goal, heuristic=zero_heuristic, weight=1.0)

def run_greedy(grid, start, goal):
    """Run Greedy (f=h, weight=0 flag)."""
    return search(grid, start, goal, heuristic=manhattan_heuristic, weight=0)

def run_weighted_astar(grid, start, goal, weight=1.5):
    """Run Weighted A* with given weight >1."""
    return search(grid, start, goal, heuristic=manhattan_heuristic, weight=weight)


if __name__ == "__main__":
   
    from grid import generate_grid, plot_grid
    grid_easy, start_easy, goal_easy = generate_grid('easy')
    grid_easy, start_easy, goal_easy = generate_grid('medium')
    grid_easy, start_easy, goal_easy = generate_grid('hard')

    path_astar, cost_astar, exp_astar, max_open_astar, time_astar = run_astar(grid_easy, start_easy, goal_easy)
    print(f"A* Path: {path_astar}, Cost: {cost_astar}, Expanded: {exp_astar}, Max OPEN: {max_open_astar}, Time: {time_astar:.4f}s")
    
    plot_grid(grid_easy, start_easy, goal_easy, path=path_astar, filename='easy_astar_path.png')
    plot_grid(grid_easy, start_easy, goal_easy, path=path_astar, filename='medium_astar_path.png')
    plot_grid(grid_easy, start_easy, goal_easy, path=path_astar, filename='hard_astar_path.png')