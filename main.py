import os
import matplotlib
matplotlib.use('Agg')  
from experiments import run_experiment_1, run_experiment_2, run_experiment_3, run_experiment_4
from visualizations import visualize_path_on_grid, visualize_experiment_comparisons
from grid import generate_grid
from astar import run_astar
import pandas as pd

def main():
    print(" Démarrage du Mini-Projet...")
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    print(" Dossiers results/ et figures/ prêts")
    
    # === EXPÉRIENCES ===
    print("\n[1/4] Comparaison UCS/Greedy/A*...")
    run_experiment_1()
    
    print("[2/4] Analyse impact ε (Markov)...")
    run_experiment_2()
    
    print("[3/4] Analyse des heuristiques...")
    run_experiment_3()
    
    print("[4/4] Weighted A*...")
    run_experiment_4()
    
   
    print("\n  Création des figures...")
    grid, start, goal = generate_grid('hard')
    path, _, _, _, _ = run_astar(grid, start, goal)
    

    visualize_path_on_grid(grid, start, goal, path, filename='hard_astar_path.png')
    
  
    df_exp1 = pd.read_csv('results/experiment_1_comparisons.csv')
    visualize_experiment_comparisons(df_exp1, metric_cols=['Nodes Expanded'], filename_prefix='exp1_nodes_')
    
    from grid import plot_grid
    plot_grid(grid, start, goal, path, filename='demo_final.png')
    
    print("\n TOUT EST TERMINÉ !")
    print("    results/  → 4 fichiers CSV")
    print("    figures/  → plusieurs images PNG")
    print("   Ouvre le dossier figures/ maintenant !")

if __name__ == "__main__":
    main()