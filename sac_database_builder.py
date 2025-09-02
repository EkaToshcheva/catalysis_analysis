"""
SAC Database Builder and Utilities
Tools for building atom and substrate property databases
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path

class SACDatabaseBuilder:
    """Build and manage databases of atom and substrate properties"""
    
    def __init__(self, database_dir: str = "sac_databases"):
        """Initialize database builder"""
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(exist_ok=True)
        
    def build_atom_database_from_csv(self, csv_path: str, output_name: str = "atom_database.json"):
        """
        Extract unique atom properties from training data
        
        Args:
            csv_path: Path to the SAC features CSV
            output_name: Name for the output database file
        """
        print("Building atom database from training data...")
        df = pd.read_csv(csv_path)
        
        # Check if 'sac_atom' column exists
        if 'sac_atom' not in df.columns:
            print("ERROR: 'sac_atom' column not found in CSV!")
            print(f"Available columns: {', '.join(df.columns[:10])}...")
            return {}
        
        # Get unique atoms
        unique_atoms = df['sac_atom'].dropna().unique()
        print(f"Found {len(unique_atoms)} unique atoms")
        
        # Atom feature columns
        atom_features = [col for col in df.columns if col.startswith('element_')]
        print(f"Found {len(atom_features)} element features")
        
        # Build database
        atom_database = {}
        
        for atom in unique_atoms:
            # Get first occurrence of this atom (properties should be consistent)
            atom_rows = df[df['sac_atom'] == atom]
            
            if len(atom_rows) == 0:
                print(f"  Warning: No data found for atom '{atom}', skipping...")
                continue
                
            atom_data = atom_rows.iloc[0]
            
            atom_properties = {}
            for feature in atom_features:
                if feature in atom_data.index:
                    value = atom_data[feature]
                    # Convert numpy types to Python types for JSON serialization
                    if pd.notna(value):
                        if isinstance(value, (np.integer, np.int64)):
                            value = int(value)
                        elif isinstance(value, (np.floating, np.float64)):
                            value = float(value)
                        atom_properties[feature] = value
            
            # Only add atom if it has properties
            if atom_properties:
                atom_database[atom] = atom_properties
            else:
                print(f"  Warning: No valid properties found for atom '{atom}', skipping...")
        
        # Save database
        output_path = self.database_dir / output_name
        with open(output_path, 'w') as f:
            json.dump(atom_database, f, indent=2)
        
        print(f"Atom database saved to {output_path}")
        print(f"Database contains {len(atom_database)} atoms with {len(atom_features)} features each")
        
        return atom_database
    
    def build_substrate_database_from_csv(self, csv_path: str, output_name: str = "substrate_database.json"):
        """
        Extract unique substrate properties from training data
        
        Args:
            csv_path: Path to the SAC features CSV
            output_name: Name for the output database file
        """
        print("Building substrate database from training data...")
        df = pd.read_csv(csv_path)
        
        # Get unique substrates
        unique_substrates = df['substrate'].unique()
        print(f"Found {len(unique_substrates)} unique substrates")
        
        # Substrate feature columns
        substrate_features = ['substrate_energy', 'substrate_energy_per_atom', 'substrate_fermi',
                            'substrate_num_atoms', 'surface_O_count', 'surface_metal_count',
                            'total_surface_atoms', 'band_gap', 'homo', 'lumo', 'cbm_energy',
                            'vbm_energy', 'is_direct_gap', 'is_metal', 'efermi', 'nelect',
                            'dos_at_efermi', 'eatom_total']
        
        # Build database
        substrate_database = {}
        
        for substrate in unique_substrates:
            # Get first occurrence of this substrate
            substrate_rows = df[df['substrate'] == substrate]
            
            if len(substrate_rows) == 0:
                print(f"  Warning: No data found for substrate '{substrate}', skipping...")
                continue
                
            substrate_data = substrate_rows.iloc[0]
            
            substrate_properties = {}
            for feature in substrate_features:
                if feature in substrate_data.index:
                    value = substrate_data[feature]
                    if pd.notna(value):
                        if isinstance(value, (np.integer, np.int64)):
                            value = int(value)
                        elif isinstance(value, (np.floating, np.float64)):
                            value = float(value)
                        elif isinstance(value, np.bool_):
                            value = bool(value)
                        substrate_properties[feature] = value
            
            # Only add substrate if it has properties
            if substrate_properties:
                substrate_database[substrate] = substrate_properties
            else:
                print(f"  Warning: No valid properties found for substrate '{substrate}', skipping...")
        
        # Save database
        output_path = self.database_dir / output_name
        with open(output_path, 'w') as f:
            json.dump(substrate_database, f, indent=2)
        
        print(f"Substrate database saved to {output_path}")
        print(f"Database contains {len(substrate_database)} substrates with up to {len(substrate_features)} features")
        
        return substrate_database
    
    def load_atom_database(self, filename: str = "atom_database.json") -> Dict:
        """Load atom database from JSON file"""
        path = self.database_dir / filename
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_substrate_database(self, filename: str = "substrate_database.json") -> Dict:
        """Load substrate database from JSON file"""
        path = self.database_dir / filename
        with open(path, 'r') as f:
            return json.load(f)
    
    def add_custom_atom(self, atom_name: str, properties: Dict, 
                       database_file: str = "atom_database.json"):
        """Add a custom atom to the database"""
        # Load existing database
        db_path = self.database_dir / database_file
        if db_path.exists():
            with open(db_path, 'r') as f:
                database = json.load(f)
        else:
            database = {}
        
        # Add new atom
        database[atom_name] = properties
        
        # Save updated database
        with open(db_path, 'w') as f:
            json.dump(database, f, indent=2)
        
        print(f"Added {atom_name} to atom database")
    
    def add_custom_substrate(self, substrate_name: str, properties: Dict,
                           database_file: str = "substrate_database.json"):
        """Add a custom substrate to the database"""
        # Load existing database
        db_path = self.database_dir / database_file
        if db_path.exists():
            with open(db_path, 'r') as f:
                database = json.load(f)
        else:
            database = {}
        
        # Add new substrate
        database[substrate_name] = properties
        
        # Save updated database
        with open(db_path, 'w') as f:
            json.dump(database, f, indent=2)
        
        print(f"Added {substrate_name} to substrate database")
    
    def get_database_statistics(self):
        """Print statistics about the databases"""
        # Atom database stats
        try:
            atom_db = self.load_atom_database()
            print(f"\nAtom Database Statistics:")
            print(f"  Total atoms: {len(atom_db)}")
            print(f"  Atoms: {', '.join(list(atom_db.keys())[:10])}...")
            
            # Feature statistics
            if atom_db:
                first_atom = list(atom_db.values())[0]
                print(f"  Features per atom: {len(first_atom)}")
                print(f"  Feature names: {', '.join(list(first_atom.keys())[:5])}...")
        except:
            print("No atom database found")
        
        # Substrate database stats
        try:
            substrate_db = self.load_substrate_database()
            print(f"\nSubstrate Database Statistics:")
            print(f"  Total substrates: {len(substrate_db)}")
            print(f"  Substrates: {', '.join(list(substrate_db.keys())[:5])}...")
            
            # Feature statistics
            if substrate_db:
                first_substrate = list(substrate_db.values())[0]
                print(f"  Features per substrate: {len(first_substrate)}")
        except:
            print("No substrate database found")


def create_screening_batch(atom_list: List[str], substrate_list: List[str],
                         atom_db_path: str, substrate_db_path: str) -> tuple:
    """
    Prepare a batch of atom-substrate combinations for screening
    
    Args:
        atom_list: List of atom symbols to screen
        substrate_list: List of substrate names to screen
        atom_db_path: Path to atom database
        substrate_db_path: Path to substrate database
        
    Returns:
        atoms_data: Dictionary of atom properties
        substrates_data: Dictionary of substrate properties
    """
    # Load databases
    with open(atom_db_path, 'r') as f:
        atom_db = json.load(f)
    
    with open(substrate_db_path, 'r') as f:
        substrate_db = json.load(f)
    
    # Extract requested atoms
    atoms_data = {}
    for atom in atom_list:
        if atom in atom_db:
            atoms_data[atom] = atom_db[atom]
        else:
            print(f"Warning: Atom '{atom}' not found in database")
    
    # Extract requested substrates
    substrates_data = {}
    for substrate in substrate_list:
        if substrate in substrate_db:
            substrates_data[substrate] = substrate_db[substrate]
        else:
            print(f"Warning: Substrate '{substrate}' not found in database")
    
    print(f"Prepared {len(atoms_data)} atoms and {len(substrates_data)} substrates for screening")
    
    return atoms_data, substrates_data


# Example periodic table data for common SAC atoms
PERIODIC_TABLE_DATA = {
    'Fe': {
        'element_electronegativity': 1.83,
        'element_ionization_energy': 7.9024,
        'element_atomic_mass': 55.845,
        'element_atomic_number': 26,
        'element_atomic_radius': 1.26,
        'element_bulk_modulus': 170.0,
        'element_electron_affinity': 0.151,
        'element_group': 8,
        'element_van_der_waals_radius': 1.94,
        'element_youngs_modulus': 211.0,
        'element_atomic_orbitals': '3d6_4s2',
        'element_block': 'd',
        'element_oxidation_states': '2,3',
        'element_valence': '3d6_4s2'
    },
    'Co': {
        'element_electronegativity': 1.88,
        'element_ionization_energy': 7.8810,
        'element_atomic_mass': 58.933,
        'element_atomic_number': 27,
        'element_atomic_radius': 1.25,
        'element_bulk_modulus': 180.0,
        'element_electron_affinity': 0.662,
        'element_group': 9,
        'element_van_der_waals_radius': 1.92,
        'element_youngs_modulus': 209.0,
        'element_atomic_orbitals': '3d7_4s2',
        'element_block': 'd',
        'element_oxidation_states': '2,3',
        'element_valence': '3d7_4s2'
    },
    'Ni': {
        'element_electronegativity': 1.91,
        'element_ionization_energy': 7.6398,
        'element_atomic_mass': 58.693,
        'element_atomic_number': 28,
        'element_atomic_radius': 1.24,
        'element_bulk_modulus': 180.0,
        'element_electron_affinity': 1.157,
        'element_group': 10,
        'element_van_der_waals_radius': 1.63,
        'element_youngs_modulus': 200.0,
        'element_atomic_orbitals': '3d8_4s2',
        'element_block': 'd',
        'element_oxidation_states': '2',
        'element_valence': '3d8_4s2'
    },
    'Cu': {
        'element_electronegativity': 1.90,
        'element_ionization_energy': 7.7264,
        'element_atomic_mass': 63.546,
        'element_atomic_number': 29,
        'element_atomic_radius': 1.28,
        'element_bulk_modulus': 140.0,
        'element_electron_affinity': 1.228,
        'element_group': 11,
        'element_van_der_waals_radius': 1.40,
        'element_youngs_modulus': 110.0,
        'element_atomic_orbitals': '3d10_4s1',
        'element_block': 'd',
        'element_oxidation_states': '1,2',
        'element_valence': '3d10_4s1'
    }
}


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("SAC DATABASE BUILDER")
    print("="*60)
    
    # Initialize builder
    builder = SACDatabaseBuilder()
    
    # Build databases from CSV
    print("\n1. Building databases from training data...")
    atom_db = builder.build_atom_database_from_csv('sac_features_consolidated.csv')
    substrate_db = builder.build_substrate_database_from_csv('sac_features_consolidated.csv')
    
    # Show statistics
    print("\n2. Database statistics:")
    builder.get_database_statistics()
    
    # Add custom atoms
    print("\n3. Adding custom atoms from periodic table...")
    for atom, properties in PERIODIC_TABLE_DATA.items():
        builder.add_custom_atom(atom, properties)
    
    # Example: Prepare screening batch
    print("\n4. Preparing screening batch...")
    atoms_to_screen = ['Fe', 'Co', 'Ni', 'Cu', 'Pt', 'Pd']
    substrates_to_screen = ['CeO2', 'TiO2', 'ZrO2']
    
    # Get available substrates
    substrate_db = builder.load_substrate_database()
    available_substrates = [s for s in substrates_to_screen if s in substrate_db]
    
    if available_substrates:
        atoms_data, substrates_data = create_screening_batch(
            atoms_to_screen, 
            available_substrates,
            'sac_databases/atom_database.json',
            'sac_databases/substrate_database.json'
        )
        
        print(f"\nReady to screen {len(atoms_data)} × {len(substrates_data)} = {len(atoms_data) * len(substrates_data)} combinations")
