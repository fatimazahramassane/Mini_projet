
import numpy as np
import networkx as nx
from collections import deque
from grid import get_neighbors, is_valid_position, DIRECTIONS


np.random.seed(42)

def compute_distances_to_goal(grid, goal):
    """
    Compute shortest path distances from all positions to goal using BFS (UCS since unit costs).
    Returns dist dict: pos -> min cost to goal.
    """
    rows, cols = grid.shape
    dist = {goal: 0}
    queue = deque([goal])
    visited = set([goal])
    
    while queue:
        current = queue.popleft()
        for neigh in get_neighbors(grid, current):
            if neigh not in visited:
                visited.add(neigh)
                dist[neigh] = dist[current] + 1
                queue.append(neigh)
    
    
    return dist

def compute_policy_from_path(grid, path, goal):
    
    if not path:
        return {}
    
    
    policy = {}
    for i in range(len(path)-1):
        current = path[i]
        next_pos = path[i+1]
        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]
        policy[current] = (dr, dc)
    policy[goal] = None

   
    dist = compute_distances_to_goal(grid, goal)
    for pos in dist:
        if pos not in policy and pos != goal:
            min_dist = min(dist[neigh] for neigh in get_neighbors(grid, pos))
            for dr, dc in DIRECTIONS:
                neigh = (pos[0] + dr, pos[1] + dc)
                if is_valid_position(grid, neigh) and dist.get(neigh, float('inf')) == min_dist:
                    policy[pos] = (dr, dc)
                    break
    return policy
def get_lateral_directions(dir):
    """Get lateral directions perpendicular to given dir."""
    dr, dc = dir
    if dr != 0:  
        return [(0, -1), (0, 1)]  
    else:  
        return [(-1, 0), (1, 0)]  

def build_transition_matrix(grid, goal, policy, epsilon=0.1):
    
    rows, cols = grid.shape
    free_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 0]
    state_list = sorted(free_positions) + ['FAIL']  
    num_states = len(state_list)
    state_to_idx = {state: idx for idx, state in enumerate(state_list)}
    
    idx_goal = state_to_idx[goal]
    idx_fail = state_to_idx['FAIL']
    
    P = np.zeros((num_states, num_states))
    
    for idx, state in enumerate(state_list):
        if state == 'FAIL' or state == goal:
            P[idx, idx] = 1.0  
            continue
        
        dir = policy[state]
        # Intended move
        next_r, next_c = state[0] + dir[0], state[1] + dir[1]
        next_pos = (next_r, next_c)
        p_intended = 1 - epsilon
        if is_valid_position(grid, next_pos):
            next_idx = state_to_idx[next_pos]
            P[idx, next_idx] += p_intended
        else:
            P[idx, idx_fail] += p_intended
        
        # Deviations
        lat_dirs = get_lateral_directions(dir)
        p_dev = epsilon / 2
        for lat_dir in lat_dirs:
            dev_r, dev_c = state[0] + lat_dir[0], state[1] + lat_dir[1]
            dev_pos = (dev_r, dev_c)
            if is_valid_position(grid, dev_pos):
                dev_idx = state_to_idx[dev_pos]
                P[idx, dev_idx] += p_dev
            else:
                P[idx, idx_fail] += p_dev
    
    # Verify stochastic
    row_sums = np.sum(P, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8), "Matrix P is not stochastic!"
    
    return P, state_to_idx, state_list

def compute_pi_n(pi0, P, n):
    """Compute pi(n) = pi0 * P^n"""
    P_n = np.linalg.matrix_power(P, n)
    pi_n = pi0 @ P_n
    return pi_n

def compute_absorption(P, state_to_idx, goal, transient_states=None):
    """
    Compute absorption probabilities and mean times.
    Absorbing: goal and 'FAIL'
    """
    idx_goal = state_to_idx[goal]
    idx_fail = state_to_idx['FAIL']
    absorbing_states = [goal, 'FAIL']
    if transient_states is None:
        transient_states = [s for s in state_to_idx if s not in absorbing_states]
    
    t_idx = [state_to_idx[s] for s in transient_states]
    a_idx = [idx_goal, idx_fail]  # Order: GOAL first, FAIL second
    
    Q = P[np.ix_(t_idx, t_idx)]
    R = P[np.ix_(t_idx, a_idx)]
    
    num_t = len(t_idx)
    I = np.eye(num_t)
    N = np.linalg.inv(I - Q)
    
    B = N @ R  # B[:,0] prob to GOAL, B[:,1] to FAIL
    mean_times = np.sum(N, axis=1)  # Mean time from each transient
    
    return B, mean_times, transient_states

def analyze_markov(P, state_list):
    """
    Analyze Markov chain: graph, classes, absorption, periodicity (option).
    Uses networkx.
    Returns dict with classes, transients, recurrents.
    """
    G = nx.DiGraph()
    num_states = len(P)
    for i in range(num_states):
        for j in range(num_states):
            if P[i, j] > 0:
                G.add_edge(state_list[i], state_list[j])
    
    # Classes: strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    
    # Transients vs recurrents: recurrents have no outgoing to other classes
    recurrents = []
    transients = []
    for scc in sccs:
        is_recurrent = True
        for node in scc:
            for succ in G.successors(node):
                if succ not in scc:
                    is_recurrent = False
                    break
            if not is_recurrent:
                break
        if is_recurrent:
            recurrents.append(scc)
        else:
            transients.append(scc)
    
    # Periodicity (option): for each recurrent class >1, gcd of cycle lengths
    periods = {}
    for rec in recurrents:
        if len(rec) == 1:
            periods[tuple(rec)] = 1  # Aperiodic
            continue
        sub_g = G.subgraph(rec)
        cycles = list(nx.simple_cycles(sub_g))
        if not cycles:
            periods[tuple(rec)] = 1
            continue
        lengths = [len(c) for c in cycles]
        gcd = np.gcd.reduce(lengths)
        periods[tuple(rec)] = gcd
    
    return {
        'classes': sccs,
        'transient_classes': transients,
        'recurrent_classes': recurrents,
        'periods': periods
    }

def simulate_markov(P, state_to_idx, state_list, start, goal, num_sim=5000, max_steps=1000):
    """
    Monte-Carlo simulation of trajectories.
    Returns empirical prob to GOAL, mean time to absorption, confidence.
    """
    idx_start = state_to_idx[start]
    idx_goal = state_to_idx[goal]
    idx_fail = state_to_idx['FAIL']
    
    successes = 0
    absorption_times = []
    
    for _ in range(num_sim):
        current = idx_start
        step = 0
        while step < max_steps:
            if current == idx_goal:
                successes += 1
                absorption_times.append(step)
                break
            elif current == idx_fail:
                absorption_times.append(step)
                break
            # Sample next
            probs = P[current]
            current = np.random.choice(len(probs), p=probs)
            step += 1
        else:
            # Timeout: count as fail
            absorption_times.append(max_steps)
    
    prob_goal = successes / num_sim
    mean_time = np.mean(absorption_times)
    # Binomial std for prob
    std_prob = np.sqrt(prob_goal * (1 - prob_goal) / num_sim) if num_sim > 0 else 0
    
    return prob_goal, mean_time, std_prob, absorption_times 


if __name__ == "__main__":
    from grid import generate_grid
    from astar import run_astar
    grid, start, goal = generate_grid('easy')
    path, _, _, _, _ = run_astar(grid, start, goal)
    policy = compute_policy_from_path(grid, path, goal)
    
    P, state_to_idx, state_list = build_transition_matrix(grid, goal, policy, epsilon=0.1)
    print(f" Test OK - P shape: {P.shape}")