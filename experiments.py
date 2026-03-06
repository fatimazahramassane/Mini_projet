
import pandas as pd
import numpy as np
from astar import run_astar, run_ucs, run_greedy, run_weighted_astar
from markov import (
    compute_policy_from_path,   
    build_transition_matrix,
    compute_absorption,
    simulate_markov,
    analyze_markov,            
)
from grid import generate_grid
import os

np.random.seed(42)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def run_experiment_1():
    """E.1: UCS vs Greedy vs A* on easy/medium/hard grids"""
    results = []
    for diff in ['easy', 'medium', 'hard']:
        grid, start, goal = generate_grid(diff)
        for algo_name, algo_func in [
            ('UCS', run_ucs),
            ('Greedy', run_greedy),
            ('A*', run_astar),
        ]:
            path, cost, expanded, max_open, exec_time = algo_func(grid, start, goal)
            results.append({
                'Grid': diff.capitalize(),
                'Algorithm': algo_name,
                'Cost': cost,
                'Nodes Expanded': expanded,
                'Max OPEN': max_open,
                'Time (s)': round(exec_time, 4)
            })
    df = pd.DataFrame(results)
    df.to_csv('results/experiment_1_comparisons.csv', index=False)
    print("=== Experiment 1 Results ===")
    print(df.to_string(index=False))
    return df

def run_experiment_2():
    """E.2: Fix A* path, policy from path, vary ε, absorption + simulation + Markov analysis"""
    epsilons = [0.0, 0.1, 0.2, 0.3]
    results = []
    for diff in ['easy', 'medium', 'hard']:
        grid, start, goal = generate_grid(diff)
        path, cost, _, _, _ = run_astar(grid, start, goal)                    
        policy = compute_policy_from_path(grid, path, goal)                  
        
        for eps in epsilons:
            P, state_to_idx, state_list = build_transition_matrix(grid, goal, policy, eps)
            
            # Phase 4 : Analyse Markov (classes de communication)
            analysis = analyze_markov(P, state_list)
            print(f"[{diff.upper()} | ε={eps}] Classes récurrentes : {analysis['recurrent_classes']}")
            
            # Absorption (exact)
            transient_states = [s for s in state_to_idx if s not in {goal, 'FAIL'}]
            B, mean_times, _ = compute_absorption(P, state_to_idx, goal, transient_states=transient_states)
            idx_trans_start = transient_states.index(start)
            prob_goal = B[idx_trans_start, 0]
            mean_time = mean_times[idx_trans_start]
            
            # Simulation Monte-Carlo
            sim_prob, sim_mean_time, std_prob, _ = simulate_markov(P, state_to_idx, state_list, start, goal, num_sim=5000)
            
            results.append({
                'Grid': diff.capitalize(),
                'ε': eps,
                'A* Cost': cost,
                'P(GOAL) Absorption': round(prob_goal, 4),
                'Mean Time Absorption': round(mean_time, 2),
                'P(GOAL) Simulation': round(sim_prob, 4),
                'Sim Mean Time': round(sim_mean_time, 2),
                'Sim Std': round(std_prob, 4)
            })
    df = pd.DataFrame(results)
    df.to_csv('results/experiment_2_epsilon_impact.csv', index=False)
    print("=== Experiment 2 Results ===")
    print(df.to_string(index=False))
    return df

def run_experiment_3():
    """E.3: h=0 vs Manhattan on 3 grids → expansions"""
    results = []
    for diff in ['easy', 'medium', 'hard']:
        grid, start, goal = generate_grid(diff)
        for h_name, algo_func in [
            ('Zero (UCS)', run_ucs),
            ('Manhattan', run_astar),
        ]:
            path, cost, expanded, max_open, t = algo_func(grid, start, goal)
            results.append({
                'Grid': diff.capitalize(),
                'Heuristic': h_name,
                'Cost': cost,
                'Nodes Expanded': expanded,
                'Max OPEN': max_open,
                'Time (s)': round(t, 4)
            })
    df = pd.DataFrame(results)
    df.to_csv('results/experiment_3_heuristics.csv', index=False)
    print("=== Experiment 3 Results ===")
    print(df.to_string(index=False))
    return df

def run_experiment_4():
    """E.4: Weighted A* (weight=1.5) vs standard A* on 3 grids"""
    results = []
    for diff in ['easy', 'medium', 'hard']:
        grid, start, goal = generate_grid(diff)
        for w, algo_name in [(1.0, 'A* (w=1.0)'), (1.5, 'Weighted A* (w=1.5)')]:
            func = run_astar if w == 1.0 else lambda g,s,go: run_weighted_astar(g,s,go, w)
            path, cost, expanded, max_open, t = func(grid, start, goal)
            results.append({
                'Grid': diff.capitalize(),
                'Algorithm': algo_name,
                'Cost': cost,
                'Nodes Expanded': expanded,
                'Max OPEN': max_open,
                'Time (s)': round(t, 4)
            })
    df = pd.DataFrame(results)
    df.to_csv('results/experiment_4_weighted_astar.csv', index=False)
    print("=== Experiment 4 Results ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print("Running all recommended experiments (Section 8 du PDF)...\n")
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    print("\n Tous les résultats sont sauvegardés dans le dossier ./results/")