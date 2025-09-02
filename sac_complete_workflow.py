"""
Complete SAC Catalyst Screening Workflow
End-to-end process from data to screening results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Import our modules
from sac_complete_system import SACCatalystScreener, train_sac_model, load_and_screen
from sac_database_builder import SACDatabaseBuilder, create_screening_batch

def run_complete_workflow(csv_path: str = None):
    """
    Run the complete workflow from raw data to screening results
    """
    # Use fixed file if available and no path specified
    if csv_path is None:
        import os
        if os.path.exists('sac_features_consolidated_fixed.csv'):
            csv_path = 'sac_features_consolidated_fixed.csv'
            print(f"Using fixed data file: {csv_path}")
        else:
            csv_path = 'sac_features_consolidated.csv'
    
    print("="*80)
    print("COMPLETE SAC CATALYST SCREENING WORKFLOW")
    print("="*80)
    
    # Step 1: Build databases
    print("\nSTEP 1: Building Property Databases")
    print("-"*60)
    
    builder = SACDatabaseBuilder()
    atom_db = builder.build_atom_database_from_csv(csv_path)
    substrate_db = builder.build_substrate_database_from_csv(csv_path)
    
    # Step 2: Train the model
    print("\nSTEP 2: Training Screening Model")
    print("-"*60)
    
    screener = train_sac_model(csv_path)
    
    # Step 3: Prepare screening candidates
    print("\nSTEP 3: Preparing Screening Candidates")
    print("-"*60)
    
    # Get all available atoms and substrates
    all_atoms = list(atom_db.keys())
    all_substrates = list(substrate_db.keys())
    
    print(f"Available atoms: {len(all_atoms)}")
    print(f"Available substrates: {len(all_substrates)}")
    
    # Select subset for demonstration (you can modify this)
    # For example: screen noble metals on oxide substrates
    noble_metals = ['Pt', 'Pd', 'Au', 'Ag', 'Ru', 'Rh', 'Ir']
    transition_metals = ['Fe', 'Co', 'Ni', 'Cu', 'Mn', 'Cr', 'V']
    
    atoms_to_screen = [a for a in noble_metals + transition_metals if a in all_atoms][:10]
    
    # Select oxide substrates
    oxide_substrates = [s for s in all_substrates if 'O' in s][:10]
    
    print(f"\nScreening {len(atoms_to_screen)} atoms on {len(oxide_substrates)} substrates")
    print(f"Total combinations: {len(atoms_to_screen) * len(oxide_substrates)}")
    
    # Step 4: Run screening
    print("\nSTEP 4: Running Catalyst Screening")
    print("-"*60)
    
    atoms_data = {atom: atom_db[atom] for atom in atoms_to_screen}
    substrates_data = {sub: substrate_db[sub] for sub in oxide_substrates}
    
    results = screener.screen_combinations(atoms_data, substrates_data)
    
    # Step 5: Analyze results
    print("\nSTEP 5: Analyzing Results")
    print("-"*60)
    
    print("\nTop 10 Candidates (Most Negative Adsorption Energy):")
    print(results.head(10).to_string(index=False))
    
    # Save results
    results.to_csv('complete_screening_results.csv', index=False)
    print("\nResults saved to 'complete_screening_results.csv'")
    
    return results, screener


def visualize_screening_results(results: pd.DataFrame, save_dir: str = "screening_plots"):
    """
    Create comprehensive visualizations of screening results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Heatmap of all combinations
    plt.figure(figsize=(12, 10))
    
    # Pivot data for heatmap
    heatmap_data = results.pivot(
        index='Atom', 
        columns='Substrate', 
        values='Predicted_Adsorption_Energy'
    )
    
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
                annot=True, fmt='.2f', cbar_kws={'label': 'Adsorption Energy (eV)'})
    plt.title('Predicted Adsorption Energy Heatmap')
    plt.tight_layout()
    plt.savefig(save_dir / 'adsorption_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top candidates bar plot
    plt.figure(figsize=(10, 8))
    
    top_20 = results.head(20).copy()
    top_20['Combination'] = top_20['Atom'] + ' on ' + top_20['Substrate']
    
    colors = ['darkgreen' if bs == 'Strong' else 'green' if bs == 'Moderate' else 'orange' 
              for bs in top_20['Binding_Strength']]
    
    plt.barh(range(len(top_20)), top_20['Predicted_Adsorption_Energy'], color=colors)
    plt.yticks(range(len(top_20)), top_20['Combination'])
    plt.xlabel('Predicted Adsorption Energy (eV)')
    plt.title('Top 20 Catalyst Candidates')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_dir / 'top_candidates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Atom performance distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Average adsorption by atom
    ax = axes[0]
    atom_avg = results.groupby('Atom')['Predicted_Adsorption_Energy'].agg(['mean', 'std'])
    atom_avg = atom_avg.sort_values('mean')
    
    ax.bar(range(len(atom_avg)), atom_avg['mean'], yerr=atom_avg['std'], capsize=5)
    ax.set_xticks(range(len(atom_avg)))
    ax.set_xticklabels(atom_avg.index, rotation=45)
    ax.set_ylabel('Average Adsorption Energy (eV)')
    ax.set_title('Average Performance by Atom')
    
    # Substrate selectivity
    ax = axes[1]
    substrate_avg = results.groupby('Substrate')['Predicted_Adsorption_Energy'].agg(['mean', 'std'])
    substrate_avg = substrate_avg.sort_values('mean')
    
    ax.bar(range(len(substrate_avg)), substrate_avg['mean'], yerr=substrate_avg['std'], capsize=5)
    ax.set_xticks(range(len(substrate_avg)))
    ax.set_xticklabels(substrate_avg.index, rotation=45, ha='right')
    ax.set_ylabel('Average Adsorption Energy (eV)')
    ax.set_title('Average Performance by Substrate')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_by_component.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Binding strength distribution
    plt.figure(figsize=(10, 6))
    
    binding_counts = results['Binding_Strength'].value_counts()
    colors = {'Strong': 'darkgreen', 'Moderate': 'green', 
              'Weak': 'orange', 'Very Weak': 'red'}
    
    plt.pie(binding_counts.values, labels=binding_counts.index, autopct='%1.1f%%',
            colors=[colors[bs] for bs in binding_counts.index])
    plt.title('Distribution of Binding Strengths')
    plt.savefig(save_dir / 'binding_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Statistical summary
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create summary statistics
    summary_text = f"""
SCREENING RESULTS SUMMARY

Total Combinations Screened: {len(results)}
Unique Atoms: {results['Atom'].nunique()}
Unique Substrates: {results['Substrate'].nunique()}

Adsorption Energy Statistics:
  Mean: {results['Predicted_Adsorption_Energy'].mean():.3f} eV
  Std Dev: {results['Predicted_Adsorption_Energy'].std():.3f} eV
  Min (Best): {results['Predicted_Adsorption_Energy'].min():.3f} eV
  Max (Worst): {results['Predicted_Adsorption_Energy'].max():.3f} eV

Binding Strength Distribution:
"""
    
    for bs, count in binding_counts.items():
        percentage = (count / len(results)) * 100
        summary_text += f"  {bs}: {count} ({percentage:.1f}%)\n"
    
    summary_text += f"""
Top 5 Atom-Substrate Combinations:
"""
    
    for idx, row in results.head(5).iterrows():
        summary_text += f"  {row['Atom']} on {row['Substrate']}: {row['Predicted_Adsorption_Energy']:.3f} eV\n"
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'screening_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization plots saved to '{save_dir}/' directory")


def export_for_dft_validation(results: pd.DataFrame, n_candidates: int = 20,
                            output_file: str = "dft_validation_candidates.json"):
    """
    Export top candidates in a format suitable for DFT validation
    """
    top_candidates = results.head(n_candidates)
    
    validation_list = []
    for _, row in top_candidates.iterrows():
        validation_list.append({
            'atom': row['Atom'],
            'substrate': row['Substrate'],
            'predicted_adsorption_energy': row['Predicted_Adsorption_Energy'],
            'binding_strength': row['Binding_Strength'],
            'priority': 'high' if row['Binding_Strength'] == 'Strong' else 'medium'
        })
    
    with open(output_file, 'w') as f:
        json.dump(validation_list, f, indent=2)
    
    print(f"\nExported {n_candidates} candidates for DFT validation to '{output_file}'")
    
    return validation_list


def compare_with_known_catalysts(results: pd.DataFrame, known_good: list):
    """
    Compare predictions with known good catalysts
    
    Args:
        results: Screening results
        known_good: List of tuples (atom, substrate) that are known good catalysts
    """
    print("\nCOMPARISON WITH KNOWN CATALYSTS")
    print("-"*60)
    
    for atom, substrate in known_good:
        match = results[(results['Atom'] == atom) & (results['Substrate'] == substrate)]
        
        if not match.empty:
            predicted = match.iloc[0]['Predicted_Adsorption_Energy']
            strength = match.iloc[0]['Binding_Strength']
            rank = match.index[0] + 1
            
            print(f"{atom} on {substrate}:")
            print(f"  Predicted: {predicted:.3f} eV ({strength})")
            print(f"  Rank: {rank} out of {len(results)}")
        else:
            print(f"{atom} on {substrate}: Not in screening set")


# Main execution
if __name__ == "__main__":
    # Check if fixed file exists and use it
    import os
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    elif os.path.exists('sac_features_consolidated_fixed.csv'):
        csv_path = 'sac_features_consolidated_fixed.csv'
        print(f"Found fixed data file, using: {csv_path}")
    else:
        csv_path = 'sac_features_consolidated.csv'
    
    # Run complete workflow
    results, screener = run_complete_workflow(csv_path=csv_path)
    
    if results is not None:
        # Create visualizations
        print("\nCreating visualizations...")
        visualize_screening_results(results)
        
        # Export for DFT validation
        validation_candidates = export_for_dft_validation(results, n_candidates=20)
        
        # Compare with known catalysts (example)
        known_good_catalysts = [
            ('Pt', 'CeO2'),  # Example known good catalyst
            ('Pd', 'TiO2'),  # Another example
        ]
        
        # Only run comparison if these are in the results
        available_atoms = results['Atom'].unique()
        available_substrates = results['Substrate'].unique()
        
        known_available = [(a, s) for a, s in known_good_catalysts 
                          if a in available_atoms and s in available_substrates]
        
        if known_available:
            compare_with_known_catalysts(results, known_available)
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review screening results in 'complete_screening_results.csv'")
        print("2. Check visualizations in 'screening_plots/' directory")
        print("3. Validate top candidates from 'dft_validation_candidates.json' with DFT")
        print("4. Use trained model for screening new atom-substrate combinations")
