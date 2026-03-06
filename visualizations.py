
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import os
from grid import plot_grid  
from markov import analyze_markov

os.makedirs('figures', exist_ok=True)
sns.set(style="whitegrid", palette="muted") 

def visualize_path_on_grid(grid, start, goal, path, filename='path_on_grid.png'):
    """
    Visualize grid with path (enhances grid.py's plot_grid).
    """
    plot_grid(grid, start, goal, path, filename)  

def visualize_transition_matrix(P, state_list, filename='heatmap_P.png'):
    """
    Heatmap of transition matrix P.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(P, ax=ax, cmap='viridis', annot=False, xticklabels=state_list, yticklabels=state_list)
    ax.set_title('Heatmap of Transition Matrix P')
    ax.set_xlabel('Next State')
    ax.set_ylabel('Current State')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap P saved to {filepath}")

def visualize_pi_evolution(pi0, P, max_n=20, state_to_idx=None, focus_states=None, filename='pi_evolution.png'):
    """
    Line plot of pi(n) evolution for focused states (e.g., GOAL, FAIL, start).
    """
    pis = [pi0]
    for n in range(1, max_n + 1):
        pi_n = pis[-1] @ P
        pis.append(pi_n)
    pis = np.array(pis)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if focus_states:
        for state in focus_states:
            idx = state_to_idx[state]
            ax.plot(range(max_n + 1), pis[:, idx], label=str(state))
    else:
        ax.plot(range(max_n + 1), pis) 
    ax.set_title('Evolution of State Probabilities π(n)')
    ax.set_xlabel('Step n')
    ax.set_ylabel('Probability')
    ax.legend()
    plt.tight_layout()
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"π(n) evolution saved to {filepath}")

def visualize_experiment_comparisons(df, group_col='Grid', metric_cols=['Nodes Expanded', 'Time (s)'], filename_prefix='exp_comp_'):
    """
    Bar charts for experiment comparisons (e.g., UCS/Greedy/A*).
    df from experiments.py CSVs.
    """
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df, x=group_col, y=metric, hue='Algorithm', ax=ax)
        ax.set_title(f'Comparison of {metric} by {group_col} and Algorithm')
        ax.legend()
        plt.tight_layout()
        filepath = os.path.join('figures', f'{filename_prefix}{metric.replace(" ", "_")}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Bar chart for {metric} saved to {filepath}")

def visualize_simulation_histogram(absorption_times, filename='sim_histogram.png'):
    """
    Histogram of absorption times from simulations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(absorption_times, bins=20, kde=True, ax=ax)
    ax.set_title('Histogram of Absorption Times (Monte-Carlo Simulations)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Simulation histogram saved to {filepath}")

def visualize_markov_graph(P, state_list, analysis, filename='markov_graph.png'):
    """
    Directed graph of Markov transitions with classes colored.
    Uses networkx and matplotlib.
    """
    G = nx.DiGraph()
    for i, state in enumerate(state_list):
        for j, p in enumerate(P[i]):
            if p > 0:
                G.add_edge(state, state_list[j], weight=p)
    
    
    color_map = []
    for node in G:
        if node == 'FAIL':
            color_map.append('red')
        elif node == state_list[0]:  
            color_map.append('green')
        else:
            # Check if transient
            is_trans = any(node in cls for cls in analysis['transient_classes'])
            color_map.append('blue' if is_trans else 'green')
    
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, font_size=8, ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    ax.set_title('Markov Chain Graph: States and Transitions')
    plt.tight_layout()
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Markov graph saved to {filepath}")


if __name__ == "__main__":
    
    from grid import generate_grid
    from astar import run_astar
    from markov import build_transition_matrix, compute_policy_from_path, compute_pi_n, simulate_markov, analyze_markov
    grid_easy, start_easy, goal_easy = generate_grid('easy')
    path, _, _, _, _ = run_astar(grid_easy, start_easy, goal_easy)
    visualize_path_on_grid(grid_easy, start_easy, goal_easy, path, 'easy_path.png')
    
    path, _, _, _, _ = run_astar(grid_easy, start_easy, goal_easy)
    policy = compute_policy_from_path(grid_easy, path, goal_easy)
    P, state_to_idx, state_list = build_transition_matrix(grid_easy, goal_easy, policy, epsilon=0.1)
    visualize_transition_matrix(P, state_list, 'easy_P_heatmap.png')
    
    pi0 = np.zeros(len(P))
    pi0[state_to_idx[start_easy]] = 1.0
    visualize_pi_evolution(pi0, P, max_n=20, state_to_idx=state_to_idx, focus_states=[start_easy, goal_easy, 'FAIL'], filename='easy_pi_evol.png')
    
    _, _, _, absorption_times = simulate_markov(P, state_to_idx, state_list, start_easy, goal_easy, num_sim=5000)
    visualize_simulation_histogram(absorption_times, 'easy_sim_hist.png')
    
    analysis = analyze_markov(P, state_list)
    visualize_markov_graph(P, state_list, analysis, 'easy_markov_graph.png')
    
    
    df_exp1 = pd.read_csv('results/experiment_1_comparisons.csv')
    visualize_experiment_comparisons(df_exp1, metric_cols=['Nodes Expanded', 'Time (s)'], filename_prefix='exp1_')