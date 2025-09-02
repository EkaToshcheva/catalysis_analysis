import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, Lars
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

"""
Complete SAC Catalyst Screening System
A production-ready tool for predicting adsorption energies using pre-adsorption features

Author: SAC ML System
Date: 2024
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

class SACCatalystScreener:
    """
    Complete system for screening Single Atom Catalysts based on pre-adsorption features
    """
    
    def __init__(self, model_dir: str = "sac_models"):
        """
        Initialize the SAC screening system
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Define feature categories
        self.atom_features = [
            'element_electronegativity', 'element_ionization_energy', 
            'element_atomic_mass', 'element_atomic_number', 
            'element_atomic_radius', 'element_bulk_modulus',
            'element_electron_affinity', 'element_group', 
            'element_van_der_waals_radius', 'element_youngs_modulus'
        ]
        
        self.substrate_features = [
            'substrate_energy', 'substrate_energy_per_atom', 'substrate_fermi',
            'substrate_num_atoms', 'surface_O_count', 'surface_metal_count',
            'total_surface_atoms', 'band_gap', 'homo', 'lumo', 'cbm_energy',
            'vbm_energy', 'is_direct_gap', 'is_metal', 'efermi', 'nelect'
        ]
        
        self.categorical_features = [
            'element_atomic_orbitals', 'element_block', 'element_ground_level',
            'element_ground_state_term', 'element_oxidation_states', 
            'element_valence', 'dos_at_efermi'
        ]
        
        # Interaction feature definitions
        self.interaction_definitions = [
            ('element_electronegativity', 'substrate_fermi'),
            ('element_ionization_energy', 'substrate_energy_per_atom'),
            ('element_atomic_radius', 'surface_metal_count'),
            ('element_electron_affinity', 'band_gap'),
            ('element_electronegativity', 'surface_O_count'),
            ('element_atomic_radius', 'total_surface_atoms')
        ]
        
        # Initialize components
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.best_model = None
        self.best_model_name = None
        
    def load_and_prepare_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data and prepare features using only pre-adsorption information
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            X: Feature matrix
            y: Target values
        """
        print("Loading data...")
        df = pd.read_csv(csv_path)
        
        # Remove rows with missing target
        df = df.dropna(subset=['adsorption_energy'])
        print(f"Data shape after cleaning: {df.shape}")
        
        # Extract target
        y = df['adsorption_energy']
        
        # Create feature matrix with only pre-adsorption features
        feature_list = self.atom_features + self.substrate_features + self.categorical_features
        available_features = [f for f in feature_list if f in df.columns]
        
        X = df[available_features].copy()
        
        # Store metadata
        self.metadata = df[['substrate', 'sac_atom']].copy() if 'sac_atom' in df.columns else None
        
        print(f"Using {len(available_features)} pre-adsorption features")
        
        return X, y
    
    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and chemistry-inspired features
        
        Args:
            X: Original feature matrix
            
        Returns:
            X_engineered: Feature matrix with additional engineered features
        """
        print("Engineering features...")
        X_eng = X.copy()
        
        # Create interaction features
        for atom_feat, substrate_feat in self.interaction_definitions:
            if atom_feat in X.columns and substrate_feat in X.columns:
                # Product feature
                feat_name = f'{atom_feat}_X_{substrate_feat}'
                X_eng[feat_name] = X[atom_feat] * X[substrate_feat]
                
                # Ratio feature
                feat_name_ratio = f'{atom_feat}_div_{substrate_feat}'
                X_eng[feat_name_ratio] = X[atom_feat] / (X[substrate_feat] + 1e-6)
        
        # Chemistry-inspired features
        if 'element_electronegativity' in X.columns and 'surface_O_count' in X.columns:
            X_eng['oxygen_affinity_score'] = (
                X['element_electronegativity'] * 
                X['surface_O_count'] / 
                (X['total_surface_atoms'] + 1)
            )
        
        if 'element_atomic_radius' in X.columns and 'substrate_num_atoms' in X.columns:
            X_eng['size_compatibility'] = (
                X['element_atomic_radius'] / 
                (X['substrate_num_atoms'] ** 0.333)
            )
        
        if 'element_electronegativity' in X.columns and 'band_gap' in X.columns:
            X_eng['electronic_mismatch'] = abs(
                X['element_electronegativity'] - X['band_gap'] / 3.0
            )
        
        print(f"Created {len(X_eng.columns) - len(X.columns)} engineered features")
        
        return X_eng
    
    def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Handle categorical variables and scale features
        
        Args:
            X: Feature matrix
            fit: Whether to fit encoders/scalers (True for training)
            
        Returns:
            X_processed: Processed feature array
        """
        X_proc = X.copy()
        
        # Handle categorical features
        for col in self.categorical_features:
            if col in X_proc.columns:
                X_proc[col] = X_proc[col].fillna('missing')
                if X_proc[col].dtype == 'object':
                    if fit:
                        le = LabelEncoder()
                        X_proc[col] = le.fit_transform(X_proc[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        if col in self.label_encoders:
                            # Handle unseen categories
                            le = self.label_encoders[col]
                            X_proc[col] = X_proc[col].apply(
                                lambda x: le.transform([str(x)])[0] 
                                if str(x) in le.classes_ else -1
                            )
        
        # Handle missing values in numeric features
        numeric_cols = X_proc.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X_proc[col].fillna(X_proc[col].median(), inplace=True)
        
        # Store feature names
        if fit:
            self.feature_names = list(X_proc.columns)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_proc)
        else:
            # Ensure same columns as training
            for col in self.feature_names:
                if col not in X_proc.columns:
                    X_proc[col] = 0
            X_proc = X_proc[self.feature_names]
            X_scaled = self.scaler.transform(X_proc)
        
        return X_scaled, X_proc
    
    def train_models(self, X: np.ndarray, y: pd.Series, X_unscaled: pd.DataFrame):
        """
        Train multiple models and select the best one
        
        Args:
            X: Scaled feature matrix
            y: Target values
            X_unscaled: Unscaled features (for tree models)
        """
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        
        # Also split unscaled data
        X_unscaled_temp, X_unscaled_test = train_test_split(
            X_unscaled, test_size=0.2, random_state=42
        )
        X_unscaled_train, X_unscaled_val = train_test_split(
            X_unscaled_temp, test_size=0.25, random_state=42
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Define expanded models
        models = {
            # Linear Models
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'BayesianRidge': BayesianRidge(),
            'HuberRegressor': HuberRegressor(epsilon=1.35),
            
            # Tree-based Models
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=100, learning_rate=1.0, random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1
            ),
            
            # Support Vector Machines
            'SVR_RBF': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'SVR_Linear': SVR(kernel='linear', C=1.0, epsilon=0.1),
            
            # Neural Network
            'MLPRegressor': MLPRegressor(
                hidden_layer_sizes=(100, 50), activation='relu',
                solver='adam', max_iter=1000, random_state=42
            ),
            
            # K-Nearest Neighbors
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance')
        }
        
        # Try to import optional libraries
        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
        except ImportError:
            print("LightGBM not available")
        
        try:
            import catboost as cb
            models['CatBoost'] = cb.CatBoostRegressor(
                iterations=200, depth=6, learning_rate=0.1,
                random_state=42, verbose=False
            )
        except ImportError:
            print("CatBoost not available")
        
        # Define which models need scaling
        needs_scaling = ['Ridge', 'Lasso', 'ElasticNet', 'BayesianRidge', 
                        'HuberRegressor', 'SVR_RBF', 'SVR_Linear', 
                        'MLPRegressor', 'KNN']
        
        # Train and evaluate models
        best_val_score = float('inf')
        results = {}
        
        print(f"\nTraining {len(models)} models...")
        for name, model in models.items():
            print(f"\n{name}:")
            
            try:
                # Use appropriate data
                if name in needs_scaling:
                    model.fit(X_train, y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_val = model.predict(X_val)
                    y_pred_test = model.predict(X_test)
                else:
                    model.fit(X_unscaled_train, y_train)
                    y_pred_train = model.predict(X_unscaled_train)
                    y_pred_val = model.predict(X_unscaled_val)
                    y_pred_test = model.predict(X_unscaled_test)
                
                # Evaluate
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_r2 = r2_score(y_test, y_pred_test)
                
                print(f"  Train RMSE: {train_rmse:.4f}")
                print(f"  Val RMSE: {val_rmse:.4f}")
                print(f"  Test RMSE: {test_rmse:.4f}")
                print(f"  Test R²: {test_r2:.4f}")
                
                # Cross-validation for robustness check
                if name in needs_scaling:
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=5, scoring='neg_mean_squared_error'
                    )
                else:
                    cv_scores = cross_val_score(
                        model, X_unscaled_train, y_train, 
                        cv=5, scoring='neg_mean_squared_error'
                    )
                
                cv_rmse = np.sqrt(-cv_scores.mean())
                print(f"  CV RMSE: {cv_rmse:.4f}")
                
                results[name] = {
                    'model': model,
                    'val_rmse': val_rmse,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                    'needs_scaling': name in needs_scaling
                }
                
                # Track best model
                if val_rmse < best_val_score:
                    best_val_score = val_rmse
                    self.best_model = model
                    self.best_model_name = name
                    self.best_model_needs_scaling = name in needs_scaling
                    
            except Exception as e:
                print(f"  Failed to train: {str(e)}")
                continue
        
        self.models = results
        print(f"\nBest model: {self.best_model_name} (Val RMSE: {best_val_score:.4f})")
        
        # Show top 5 models
        print("\nTop 5 models by validation RMSE:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['val_rmse'])
        for i, (name, metrics) in enumerate(sorted_models[:5]):
            print(f"{i+1}. {name}: Val RMSE={metrics['val_rmse']:.4f}, Test R²={metrics['test_r2']:.4f}")
        
        # Feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 15 most important features ({self.best_model_name}):")
            for idx, row in feature_importance.head(15).iterrows():
                print(f"  {row['feature']:<40} {row['importance']:.4f}")
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        save_dict = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'best_model_needs_scaling': getattr(self, 'best_model_needs_scaling', False),
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'atom_features': self.atom_features,
            'substrate_features': self.substrate_features,
            'interaction_definitions': self.interaction_definitions,
            'models_performance': {k: {
                'val_rmse': v['val_rmse'],
                'test_rmse': v['test_rmse'],
                'test_r2': v['test_r2']
            } for k, v in self.models.items()}
        }
        
        save_path = self.model_dir / 'sac_screening_model.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"\nModel saved to {save_path}")
        
        # Also save feature definitions for reference
        feature_info = {
            'atom_features': self.atom_features,
            'substrate_features': self.substrate_features,
            'categorical_features': self.categorical_features,
            'interaction_definitions': self.interaction_definitions,
            'all_features': self.feature_names
        }
        
        with open(self.model_dir / 'feature_definitions.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
    
    def load_model(self):
        """Load a previously trained model"""
        load_path = self.model_dir / 'sac_screening_model.pkl'
        
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model found at {load_path}")
        
        with open(load_path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.best_model = save_dict['best_model']
        self.best_model_name = save_dict['best_model_name']
        self.best_model_needs_scaling = save_dict.get('best_model_needs_scaling', False)
        self.scaler = save_dict['scaler']
        self.label_encoders = save_dict['label_encoders']
        self.feature_names = save_dict['feature_names']
        
        print(f"Model loaded: {self.best_model_name}")
        if hasattr(self, 'best_model_needs_scaling'):
            print(f"Model requires scaling: {self.best_model_needs_scaling}")
        
    def predict_single(self, atom_properties: Dict, substrate_properties: Dict) -> float:
        """
        Predict adsorption energy for a single atom-substrate pair
        
        Args:
            atom_properties: Dictionary of atom properties
            substrate_properties: Dictionary of substrate properties
            
        Returns:
            Predicted adsorption energy
        """
        # Create feature vector
        feature_dict = {}
        feature_dict.update(atom_properties)
        feature_dict.update(substrate_properties)
        
        # Create dataframe
        X = pd.DataFrame([feature_dict])
        
        # Engineer features
        X = self.engineer_features(X)
        
        # Preprocess
        X_scaled, X_unscaled = self.preprocess_features(X, fit=False)
        
        # Predict based on model type
        if hasattr(self, 'best_model_needs_scaling') and self.best_model_needs_scaling:
            prediction = self.best_model.predict(X_scaled)[0]
        else:
            # Tree models use unscaled data
            prediction = self.best_model.predict(X_unscaled)[0]
        
        return prediction
    
    def screen_combinations(self, 
                          atoms_data: Dict[str, Dict],
                          substrates_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Screen multiple atom-substrate combinations
        
        Args:
            atoms_data: Dictionary of atom names to properties
            substrates_data: Dictionary of substrate names to properties
            
        Returns:
            DataFrame with screening results sorted by predicted adsorption
        """
        results = []
        
        total = len(atoms_data) * len(substrates_data)
        print(f"\nScreening {total} combinations...")
        
        for i, (atom_name, atom_props) in enumerate(atoms_data.items()):
            for j, (substrate_name, substrate_props) in enumerate(substrates_data.items()):
                try:
                    prediction = self.predict_single(atom_props, substrate_props)
                    
                    results.append({
                        'Atom': atom_name,
                        'Substrate': substrate_name,
                        'Predicted_Adsorption_Energy': prediction,
                        'Binding_Strength': self._classify_binding(prediction)
                    })
                    
                except Exception as e:
                    print(f"Error for {atom_name}-{substrate_name}: {e}")
        
        # Create sorted dataframe
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Predicted_Adsorption_Energy')
        
        return results_df
    
    def _classify_binding(self, adsorption_energy: float) -> str:
        """Classify binding strength based on adsorption energy"""
        if adsorption_energy < -2.0:
            return "Strong"
        elif adsorption_energy < -1.0:
            return "Moderate"
        elif adsorption_energy < -0.3:
            return "Weak"
        else:
            return "Very Weak"


# Main execution functions
def train_sac_model(csv_path: str, model_dir: str = "sac_models"):
    """
    Train the SAC screening model from scratch
    
    Args:
        csv_path: Path to the training data CSV
        model_dir: Directory to save the model
    """
    print("="*60)
    print("SAC CATALYST SCREENING MODEL TRAINING")
    print("="*60)
    
    # Initialize screener
    screener = SACCatalystScreener(model_dir)
    
    # Load and prepare data
    X, y = screener.load_and_prepare_data(csv_path)
    
    # Engineer features
    X_engineered = screener.engineer_features(X)
    
    # Preprocess features
    X_scaled, X_unscaled = screener.preprocess_features(X_engineered, fit=True)
    
    # Train models
    screener.train_models(X_scaled, y, X_unscaled)
    
    # Save model
    screener.save_model()
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    
    return screener


def load_and_screen(atoms_data: Dict[str, Dict], 
                   substrates_data: Dict[str, Dict],
                   model_dir: str = "sac_models") -> pd.DataFrame:
    """
    Load a trained model and screen atom-substrate combinations
    
    Args:
        atoms_data: Dictionary of atom properties
        substrates_data: Dictionary of substrate properties
        model_dir: Directory containing the saved model
        
    Returns:
        DataFrame with screening results
    """
    # Initialize and load model
    screener = SACCatalystScreener(model_dir)
    screener.load_model()
    
    # Screen combinations
    results = screener.screen_combinations(atoms_data, substrates_data)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example 1: Train the model
    print("Training SAC screening model...")
    screener = train_sac_model('sac_features_consolidated.csv')
    
    # Example 2: Use the model for screening
    print("\n" + "="*60)
    print("CATALYST SCREENING EXAMPLE")
    print("="*60)
    
    # Example atom properties (you would load these from a database)
    example_atoms = {
        'Pt': {
            'element_electronegativity': 2.28,
            'element_ionization_energy': 8.9587,
            'element_atomic_mass': 195.084,
            'element_atomic_number': 78,
            'element_atomic_radius': 1.39,
            'element_bulk_modulus': 230.0,
            'element_electron_affinity': 2.128,
            'element_group': 10,
            'element_van_der_waals_radius': 1.75,
            'element_youngs_modulus': 168.0
        },
        'Pd': {
            'element_electronegativity': 2.20,
            'element_ionization_energy': 8.3369,
            'element_atomic_mass': 106.42,
            'element_atomic_number': 46,
            'element_atomic_radius': 1.37,
            'element_bulk_modulus': 180.0,
            'element_electron_affinity': 0.557,
            'element_group': 10,
            'element_van_der_waals_radius': 1.63,
            'element_youngs_modulus': 121.0
        }
    }
    
    # Example substrate properties (from DFT calculations on clean surfaces)
    example_substrates = {
        'TiO2_rutile': {
            'substrate_energy': -234.567,
            'substrate_energy_per_atom': -7.819,
            'substrate_fermi': -5.234,
            'substrate_num_atoms': 30,
            'surface_O_count': 12,
            'surface_metal_count': 6,
            'total_surface_atoms': 18,
            'band_gap': 3.0,
            'homo': -7.234,
            'lumo': -4.234,
            'cbm_energy': -4.234,
            'vbm_energy': -7.234,
            'is_direct_gap': 1,
            'is_metal': 0,
            'efermi': -5.234,
            'nelect': 240.0
        }
    }
    
    # Screen combinations
    results = load_and_screen(example_atoms, example_substrates)
    
    print("\nScreening Results:")
    print(results)
    
    # Save results
    results.to_csv('screening_results.csv', index=False)
    print("\nResults saved to 'screening_results.csv'")
