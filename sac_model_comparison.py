"""
SAC Model Comparison Visual Analysis
Comprehensive visual comparison of the top 6 models for SAC catalyst screening
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec

# Import the SAC system
from sac_complete_system import SACCatalystScreener

class SACModelVisualizer:
    """Visual analysis tool for comparing SAC screening models"""
    
    def __init__(self, screener: SACCatalystScreener = None, 
                 model_dir: str = "sac_models",
                 output_dir: str = "model_comparison_plots"):
        """
        Initialize the visualizer
        
        Args:
            screener: Trained SACCatalystScreener object (if None, will load from model_dir)
            model_dir: Directory containing saved models
            output_dir: Directory to save plots
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Use provided screener or load from file
        if screener is not None:
            self.screener = screener
            print("Using provided trained screener")
        else:
            # Load from saved model
            self.screener = SACCatalystScreener(model_dir)
            self.screener.load_model()
            print(f"Loaded screener from {model_dir}")
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def load_data_for_evaluation(self, csv_path: str):
        """Load and prepare data matching the trained model's preprocessing"""
        print("Loading data for evaluation...")
        
        # Load data using screener's method
        X, y = self.screener.load_and_prepare_data(csv_path)
        
        # Engineer features
        X_engineered = self.screener.engineer_features(X)
        
        # Preprocess (using fitted preprocessors)
        X_scaled, X_unscaled = self.screener.preprocess_features(X_engineered, fit=False)
        
        # Split data with same random state as training
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
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
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        self.X_unscaled_train = X_unscaled_train
        self.X_unscaled_val = X_unscaled_val
        self.X_unscaled_test = X_unscaled_test
        
        print(f"Data loaded: {len(X)} samples, {X_scaled.shape[1]} features")
        
    def get_top_models(self, n_models: int = 6):
        """Get the top n models from already trained models"""
        print(f"\nSelecting top {n_models} models from trained set...")
        
        # Get models from screener
        if not hasattr(self.screener, 'models') or not self.screener.models:
            raise ValueError("No trained models found in screener! Make sure models are trained first.")
        
        # Sort models by validation RMSE
        sorted_models = sorted(
            self.screener.models.items(), 
            key=lambda x: x[1]['val_rmse']
        )
        
        # Get top n models
        self.top_models = dict(sorted_models[:n_models])
        
        print(f"\nTop {n_models} models selected:")
        for i, (name, metrics) in enumerate(self.top_models.items()):
            print(f"{i+1}. {name}: Val RMSE = {metrics['val_rmse']:.4f}, Test R² = {metrics['test_r2']:.4f}")
        
        # Also print which model was selected as best
        print(f"\nBest model overall: {self.screener.best_model_name}")
        
        return self.top_models
    
    def plot_performance_comparison(self):
        """Create bar plots comparing model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        
        model_names = list(self.top_models.keys())
        
        # 1. RMSE Comparison
        ax = axes[0, 0]
        val_rmse = [self.top_models[m]['val_rmse'] for m in model_names]
        test_rmse = [self.top_models[m]['test_rmse'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_rmse, width, label='Validation RMSE', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
        
        # Highlight best model
        best_idx = model_names.index(self.screener.best_model_name) if self.screener.best_model_name in model_names else -1
        if best_idx >= 0:
            bars1[best_idx].set_color('darkgreen')
            bars2[best_idx].set_color('darkgreen')
            bars1[best_idx].set_edgecolor('black')
            bars2[best_idx].set_edgecolor('black')
            bars1[best_idx].set_linewidth(2)
            bars2[best_idx].set_linewidth(2)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('RMSE (eV)')
        ax.set_title('Root Mean Square Error Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (v, t) in enumerate(zip(val_rmse, test_rmse)):
            ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, t + 0.01, f'{t:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. R² Score Comparison
        ax = axes[0, 1]
        test_r2 = [self.top_models[m]['test_r2'] for m in model_names]
        
        bars = ax.bar(model_names, test_r2, alpha=0.8, color='green')
        
        # Highlight best model
        if best_idx >= 0:
            bars[best_idx].set_color('darkgreen')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_title('Test Set R² Score Comparison')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(min(test_r2) * 0.95, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, r2 in zip(bars, test_r2):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{r2:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. MAE Comparison
        ax = axes[1, 0]
        mae_scores = []
        for name, model_info in self.top_models.items():
            model = model_info['model']
            if model_info['needs_scaling']:
                y_pred = model.predict(self.X_test)
            else:
                y_pred = model.predict(self.X_unscaled_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            mae_scores.append(mae)
        
        bars = ax.bar(model_names, mae_scores, alpha=0.8, color='orange')
        
        # Highlight best model
        if best_idx >= 0:
            bars[best_idx].set_color('darkorange')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE (eV)')
        ax.set_title('Mean Absolute Error Comparison')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mae in zip(bars, mae_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{mae:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Training Time Proxy (Model Complexity)
        ax = axes[1, 1]
        complexity_proxy = {
            'Ridge': 1, 'Lasso': 1, 'ElasticNet': 1.5, 'BayesianRidge': 2,
            'HuberRegressor': 1.5, 'RandomForest': 5, 'ExtraTrees': 5,
            'GradientBoosting': 6, 'AdaBoost': 4, 'XGBoost': 6,
            'SVR_RBF': 3, 'SVR_Linear': 2, 'MLPRegressor': 4,
            'KNN': 1.5, 'LightGBM': 5, 'CatBoost': 6, 'Lars': 1
        }
        
        complexities = [complexity_proxy.get(m, 3) for m in model_names]
        bars = ax.bar(model_names, complexities, alpha=0.8, color='purple')
        
        if best_idx >= 0:
            bars[best_idx].set_color('darkviolet')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Relative Complexity')
        ax.set_title('Model Complexity (Training Cost Proxy)')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add note about best model
        if self.screener.best_model_name in model_names:
            fig.text(0.5, 0.01, f'Best Model (highlighted): {self.screener.best_model_name}', 
                    ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_comparison.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Performance comparison plot saved (PNG & EPS).")
    
    def plot_predictions_scatter(self):
        """Create scatter plots of predictions vs actual values"""
        n_models = len(self.top_models)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('Predictions vs Actual Values (Test Set)', fontsize=16, y=1.02)
        
        axes_flat = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, model_info) in enumerate(self.top_models.items()):
            ax = axes_flat[idx]
            model = model_info['model']
            
            # Get predictions
            if model_info['needs_scaling']:
                y_pred = model.predict(self.X_test)
            else:
                y_pred = model.predict(self.X_unscaled_test)
            
            # Create scatter plot
            scatter = ax.scatter(self.y_test, y_pred, alpha=0.5, s=30)
            
            # Highlight if this is the best model
            if name == self.screener.best_model_name:
                scatter.set_edgecolors('red')
                scatter.set_linewidths(1)
                ax.set_title(f'{name} (BEST MODEL)\nRMSE: {model_info["test_rmse"]:.3f}, R²: {model_info["test_r2"]:.3f}', 
                           fontweight='bold')
            else:
                ax.set_title(f'{name}\nRMSE: {model_info["test_rmse"]:.3f}, R²: {model_info["test_r2"]:.3f}')
            
            # Add diagonal line
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual Adsorption Energy (eV)')
            ax.set_ylabel('Predicted Adsorption Energy (eV)')
            ax.grid(True, alpha=0.3)
            
            # Add error bounds
            rmse = model_info['test_rmse']
            ax.fill_between([min_val, max_val], 
                           [min_val - rmse, max_val - rmse],
                           [min_val + rmse, max_val + rmse],
                           alpha=0.1, color='gray', label=f'±RMSE')
        
        # Hide extra subplots
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictions_scatter.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'predictions_scatter.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Predictions scatter plot saved (PNG & EPS).")
    
    def plot_residual_analysis(self):
        """Create residual plots for model diagnostics"""
        n_models = min(len(self.top_models), 6)  # Limit to 6 for readability
        
        fig = plt.figure(figsize=(16, 4*n_models))
        gs = gridspec.GridSpec(n_models, 3, figure=fig)
        fig.suptitle('Residual Analysis', fontsize=16, y=1.002)
        
        for idx, (name, model_info) in enumerate(list(self.top_models.items())[:n_models]):
            model = model_info['model']
            
            # Get predictions
            if model_info['needs_scaling']:
                y_pred_test = model.predict(self.X_test)
                y_pred_train = model.predict(self.X_train)
            else:
                y_pred_test = model.predict(self.X_unscaled_test)
                y_pred_train = model.predict(self.X_unscaled_train)
            
            residuals_test = self.y_test - y_pred_test
            residuals_train = self.y_train - y_pred_train
            
            # 1. Residuals vs Predicted
            ax1 = fig.add_subplot(gs[idx, 0])
            ax1.scatter(y_pred_test, residuals_test, alpha=0.5, s=30, label='Test')
            ax1.axhline(y=0, color='r', linestyle='--', lw=2)
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            
            if name == self.screener.best_model_name:
                ax1.set_title(f'{name} (BEST) - Residuals vs Predicted', fontweight='bold')
            else:
                ax1.set_title(f'{name} - Residuals vs Predicted')
            
            ax1.grid(True, alpha=0.3)
            
            # Add ±1 std bounds
            std_res = np.std(residuals_test)
            ax1.axhline(y=std_res, color='orange', linestyle=':', alpha=0.7)
            ax1.axhline(y=-std_res, color='orange', linestyle=':', alpha=0.7)
            
            # 2. Q-Q plot
            ax2 = fig.add_subplot(gs[idx, 1])
            from scipy import stats
            stats.probplot(residuals_test, dist="norm", plot=ax2)
            ax2.set_title(f'{name} - Normal Q-Q Plot')
            ax2.grid(True, alpha=0.3)
            
            # 3. Residual histogram
            ax3 = fig.add_subplot(gs[idx, 2])
            ax3.hist(residuals_test, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(x=0, color='r', linestyle='--', lw=2)
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'{name} - Residual Distribution')
            
            # Add normal distribution overlay
            mu, std = stats.norm.fit(residuals_test)
            x = np.linspace(residuals_test.min(), residuals_test.max(), 100)
            ax3.plot(x, stats.norm.pdf(x, mu, std) * len(residuals_test) * (residuals_test.max() - residuals_test.min()) / 30,
                    'r-', lw=2, label=f'Normal(μ={mu:.3f}, σ={std:.3f})')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'residual_analysis.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Residual analysis plot saved (PNG & EPS).")
    
    def plot_feature_importance(self):
        """Compare feature importance across models that support it"""
        tree_models = {}
        for name, model_info in self.top_models.items():
            if hasattr(model_info['model'], 'feature_importances_'):
                tree_models[name] = model_info
        
        if not tree_models:
            print("No models with feature importance found.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top features
        all_importances = {}
        for name, model_info in tree_models.items():
            importances = model_info['model'].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.screener.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            all_importances[name] = feature_importance
        
        # Get top 20 features by average importance
        avg_importance = pd.DataFrame()
        for name, df in all_importances.items():
            if avg_importance.empty:
                avg_importance = df.set_index('feature')[['importance']].rename(columns={'importance': name})
            else:
                avg_importance[name] = df.set_index('feature')['importance']
        
        avg_importance['mean'] = avg_importance.mean(axis=1)
        top_features = avg_importance.nlargest(20, 'mean').index.tolist()
        
        # Plot comparison
        x = np.arange(len(top_features))
        width = 0.8 / len(tree_models)
        
        for i, (name, _) in enumerate(tree_models.items()):
            importances = []
            for feat in top_features:
                imp_df = all_importances[name]
                imp = imp_df[imp_df['feature'] == feat]['importance'].values
                importances.append(imp[0] if len(imp) > 0 else 0)
            
            bars = ax.bar(x + i * width, importances, width, label=name, alpha=0.8)
            
            # Highlight best model
            if name == self.screener.best_model_name:
                for bar in bars:
                    bar.set_edgecolor('red')
                    bar.set_linewidth(2)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance Comparison (Top 20 Features)')
        ax.set_xticks(x + width * (len(tree_models) - 1) / 2)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'feature_importance_comparison.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Feature importance comparison plot saved (PNG & EPS).")
    
    def plot_cross_validation_scores(self):
        """Plot cross-validation score distributions"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cv_results = []
        model_names = []
        
        print("Performing cross-validation for visualization...")
        for name, model_info in self.top_models.items():
            model_class = type(model_info['model'])
            model_params = model_info['model'].get_params()
            
            # Create new instance with same parameters
            new_model = model_class(**model_params)
            
            # Perform cross-validation
            if model_info['needs_scaling']:
                cv_scores = cross_val_score(
                    new_model, self.X_train, self.y_train, 
                    cv=5, scoring='neg_mean_squared_error'
                )
            else:
                cv_scores = cross_val_score(
                    new_model, self.X_unscaled_train, self.y_train, 
                    cv=5, scoring='neg_mean_squared_error'
                )
            
            cv_rmse = np.sqrt(-cv_scores)
            cv_results.extend(cv_rmse)
            model_names.extend([name] * len(cv_rmse))
        
        # Create violin plot
        cv_df = pd.DataFrame({
            'Model': model_names,
            'CV RMSE': cv_results
        })
        
        # Create custom colors - highlight best model
        palette = []
        unique_models = cv_df['Model'].unique()
        for model in unique_models:
            if model == self.screener.best_model_name:
                palette.append('darkgreen')
            else:
                palette.append('lightblue')
        
        sns.violinplot(data=cv_df, x='Model', y='CV RMSE', ax=ax, palette=palette)
        ax.set_xlabel('Model')
        ax.set_ylabel('Cross-Validation RMSE (eV)')
        ax.set_title('Cross-Validation Score Distribution (5-Fold CV)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean lines
        for i, model in enumerate(unique_models):
            model_scores = cv_df[cv_df['Model'] == model]['CV RMSE']
            mean_score = model_scores.mean()
            ax.hlines(mean_score, i-0.4, i+0.4, colors='red', linestyles='dashed', 
                     label='Mean' if i == 0 else "", linewidth=2)
        
        if self.screener.best_model_name in unique_models:
            ax.text(0.02, 0.98, f'Best Model: {self.screener.best_model_name}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cv_scores_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'cv_scores_distribution.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Cross-validation scores plot saved (PNG & EPS).")
    
    def plot_prediction_distributions(self):
        """Plot distribution of predictions for each model"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Collect all predictions
        predictions_dict = {}
        for name, model_info in self.top_models.items():
            model = model_info['model']
            if model_info['needs_scaling']:
                y_pred = model.predict(self.X_test)
            else:
                y_pred = model.predict(self.X_unscaled_test)
            predictions_dict[name] = y_pred
        
        # 1. Overlaid histograms
        ax = axes[0]
        for name, predictions in predictions_dict.items():
            if name == self.screener.best_model_name:
                ax.hist(predictions, bins=30, alpha=0.7, label=f'{name} (BEST)', 
                       density=True, edgecolor='black', linewidth=2)
            else:
                ax.hist(predictions, bins=30, alpha=0.5, label=name, density=True)
        
        ax.hist(self.y_test, bins=30, alpha=0.7, label='Actual', 
                color='black', histtype='step', linewidth=2, density=True)
        
        ax.set_xlabel('Adsorption Energy (eV)')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Box plots
        ax = axes[1]
        pred_data = [predictions for predictions in predictions_dict.values()]
        pred_labels = list(predictions_dict.keys())
        pred_data.append(self.y_test.values)
        pred_labels.append('Actual')
        
        # Create colors
        colors = []
        for label in pred_labels:
            if label == self.screener.best_model_name:
                colors.append('darkgreen')
            elif label == 'Actual':
                colors.append('lightgray')
            else:
                colors.append('lightblue')
        
        box_plot = ax.boxplot(pred_data, labels=pred_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Adsorption Energy (eV)')
        ax.set_title('Prediction Range Comparison')
        ax.set_xticklabels(pred_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'prediction_distributions.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Prediction distributions plot saved (PNG & EPS).")
    
    def plot_error_analysis(self):
        """Analyze errors by different data characteristics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Error Analysis by Data Characteristics', fontsize=16, y=1.02)
        
        # Get best model predictions
        best_model_name = self.screener.best_model_name
        if best_model_name not in self.top_models:
            # Use first model if best model not in top models
            best_model_name = list(self.top_models.keys())[0]
            print(f"Note: Best model {self.screener.best_model_name} not in top 6, using {best_model_name} for error analysis")
        
        best_model_info = self.top_models[best_model_name]
        model = best_model_info['model']
        
        if best_model_info['needs_scaling']:
            y_pred = model.predict(self.X_test)
        else:
            y_pred = model.predict(self.X_unscaled_test)
        
        errors = np.abs(self.y_test - y_pred)
        
        # 1. Error vs Actual Value
        ax = axes[0, 0]
        scatter = ax.scatter(self.y_test, errors, alpha=0.5, c=errors, cmap='viridis')
        ax.set_xlabel('Actual Adsorption Energy (eV)')
        ax.set_ylabel('Absolute Error (eV)')
        ax.set_title(f'Error vs Actual Value ({best_model_name})')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Error (eV)')
        
        # Add trend line
        z = np.polyfit(self.y_test, errors, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(self.y_test.min(), self.y_test.max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, lw=2)
        
        # 2. Error distribution by binding strength
        ax = axes[0, 1]
        binding_categories = pd.cut(self.y_test, 
                                  bins=[-np.inf, -2.0, -1.0, -0.3, np.inf],
                                  labels=['Strong', 'Moderate', 'Weak', 'Very Weak'])
        
        error_df = pd.DataFrame({
            'Binding': binding_categories,
            'Error': errors
        })
        
        sns.boxplot(data=error_df, x='Binding', y='Error', ax=ax)
        ax.set_xlabel('Binding Strength')
        ax.set_ylabel('Absolute Error (eV)')
        ax.set_title('Error Distribution by Binding Strength')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Cumulative error distribution
        ax = axes[1, 0]
        for i, (name, model_info) in enumerate(self.top_models.items()):
            model = model_info['model']
            if model_info['needs_scaling']:
                y_pred = model.predict(self.X_test)
            else:
                y_pred = model.predict(self.X_unscaled_test)
            
            errors = np.abs(self.y_test - y_pred)
            sorted_errors = np.sort(errors)
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
            
            if name == self.screener.best_model_name:
                ax.plot(sorted_errors, cumulative, label=f'{name} (BEST)', lw=3, color='darkgreen')
            else:
                ax.plot(sorted_errors, cumulative, label=name, lw=2, alpha=0.7)
        
        ax.set_xlabel('Absolute Error (eV)')
        ax.set_ylabel('Cumulative Percentage (%)')
        ax.set_title('Cumulative Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        
        # Add 90% error line
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90%')
        
        # 4. Model agreement analysis
        ax = axes[1, 1]
        all_predictions = []
        for name, model_info in self.top_models.items():
            model = model_info['model']
            if model_info['needs_scaling']:
                y_pred = model.predict(self.X_test)
            else:
                y_pred = model.predict(self.X_unscaled_test)
            all_predictions.append(y_pred)
        
        all_predictions = np.array(all_predictions)
        pred_std = np.std(all_predictions, axis=0)
        pred_mean = np.mean(all_predictions, axis=0)
        
        scatter = ax.scatter(pred_mean, pred_std, alpha=0.5, c=np.abs(self.y_test - pred_mean), cmap='plasma')
        ax.set_xlabel('Mean Prediction (eV)')
        ax.set_ylabel('Prediction Std Dev (eV)')
        ax.set_title('Model Agreement Analysis')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Actual Error (eV)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'error_analysis.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Error analysis plot saved (PNG & EPS).")
    
    def create_summary_report(self):
        """Create a summary report with key statistics"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('SAC Model Comparison Summary Report', fontsize=18, y=0.98)
        
        # 1. Model ranking table
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('tight')
        ax1.axis('off')
        
        # Create summary data
        summary_data = []
        for rank, (name, model_info) in enumerate(self.top_models.items(), 1):
            model = model_info['model']
            if model_info['needs_scaling']:
                y_pred = model.predict(self.X_test)
            else:
                y_pred = model.predict(self.X_unscaled_test)
            
            mae = mean_absolute_error(self.y_test, y_pred)
            
            # Add star if best model
            display_name = f"{name} ⭐" if name == self.screener.best_model_name else name
            
            summary_data.append([
                rank,
                display_name,
                f"{model_info['val_rmse']:.4f}",
                f"{model_info['test_rmse']:.4f}",
                f"{mae:.4f}",
                f"{model_info['test_r2']:.4f}"
            ])
        
        table = ax1.table(cellText=summary_data,
                         colLabels=['Rank', 'Model', 'Val RMSE', 'Test RMSE', 'MAE', 'R²'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.1, 0.3, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color code the best model
        for i in range(1, len(summary_data) + 1):
            model_name = summary_data[i-1][1]
            if '⭐' in model_name:  # Best model
                for j in range(6):
                    table[(i, j)].set_facecolor('#90EE90')
                    table[(i, j)].set_text_props(weight='bold')
        
        ax1.set_title('Model Performance Ranking', fontsize=14, pad=20)
        
        # 2. Best model details
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        best_name = self.screener.best_model_name
        if best_name in self.top_models:
            best_info = self.top_models[best_name]
        else:
            # If best model not in top 6, show it anyway
            best_info = self.screener.models[best_name]
        
        details_text = f"""
BEST MODEL: {best_name}

Performance Metrics:
  Val RMSE: {best_info['val_rmse']:.4f} eV
  Test RMSE: {best_info['test_rmse']:.4f} eV
  Test R²: {best_info['test_r2']:.4f}

Model Type:
  {'Tree-based' if best_name in ['RandomForest', 'XGBoost', 'GradientBoosting', 'ExtraTrees', 'LightGBM', 'CatBoost'] else 'Linear/Other'}

Requires Scaling: 
  {'Yes' if best_info.get('needs_scaling', False) else 'No'}

Training Info:
  Total models trained: {len(self.screener.models)}
  Best model rank: #{list(dict(sorted(self.screener.models.items(), key=lambda x: x[1]['val_rmse'])).keys()).index(best_name) + 1}
"""
        
        ax2.text(0.1, 0.5, details_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Performance radar chart
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        
        # Prepare data for radar chart
        categories = ['RMSE⁻¹', 'R²', 'MAE⁻¹', 'Stability', 'Speed']
        
        # Normalize metrics for radar chart (show top 3 models including best)
        models_to_show = []
        if best_name not in list(self.top_models.keys())[:3]:
            # Include best model if not in top 3
            models_to_show.append(best_name)
            models_to_show.extend(list(self.top_models.keys())[:2])
        else:
            models_to_show = list(self.top_models.keys())[:3]
        
        for name in models_to_show:
            if name in self.top_models:
                model_info = self.top_models[name]
            else:
                model_info = self.screener.models[name]
                
            model = model_info['model']
            
            # Calculate MAE if needed
            if name in self.top_models:
                if model_info['needs_scaling']:
                    y_pred = model.predict(self.X_test)
                else:
                    y_pred = model.predict(self.X_unscaled_test)
                mae = mean_absolute_error(self.y_test, y_pred)
            else:
                mae = 0.3  # Default for models not in top 6
            
            # Create normalized scores (0-1 scale, higher is better)
            scores = [
                1 / (1 + model_info['test_rmse']),  # RMSE inverse
                model_info['test_r2'],               # R²
                1 / (1 + mae),                       # MAE inverse
                0.8,                                 # Stability placeholder
                0.5                                  # Speed placeholder
            ]
            
            # Plot
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            scores += scores[:1]  # Complete the circle
            angles += angles[:1]
            
            if name == best_name:
                ax3.plot(angles, scores, 'o-', linewidth=3, label=f'{name} (BEST)', color='darkgreen')
                ax3.fill(angles, scores, alpha=0.3, color='darkgreen')
            else:
                ax3.plot(angles, scores, 'o-', linewidth=2, label=name, alpha=0.7)
                ax3.fill(angles, scores, alpha=0.15)
        
        ax3.set_theta_offset(np.pi / 2)
        ax3.set_theta_direction(-1)
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1)
        ax3.set_title('Multi-Metric Comparison', y=1.08)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax3.grid(True)
        
        # 4. Prediction scatter for best model
        ax4 = fig.add_subplot(gs[1, 1:])
        
        # Get best model and make predictions
        if best_name in self.top_models:
            best_model = self.top_models[best_name]['model']
            needs_scaling = self.top_models[best_name]['needs_scaling']
        else:
            best_model = self.screener.models[best_name]['model']
            needs_scaling = self.screener.models[best_name]['needs_scaling']
        
        if needs_scaling:
            y_pred = best_model.predict(self.X_test)
        else:
            y_pred = best_model.predict(self.X_unscaled_test)
        
        ax4.scatter(self.y_test, y_pred, alpha=0.6, s=50)
        
        # Add diagonal line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Add statistics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        ax4.text(0.05, 0.95, f'RMSE: {rmse:.3f} eV\nR²: {r2:.3f}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('Actual Adsorption Energy (eV)')
        ax4.set_ylabel('Predicted Adsorption Energy (eV)')
        ax4.set_title(f'Best Model ({best_name}) - Test Set Predictions')
        ax4.grid(True, alpha=0.3)
        
        # 5. Residual histogram for best model
        ax5 = fig.add_subplot(gs[2, :])
        
        residuals = self.y_test - y_pred
        
        n, bins, patches = ax5.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
        
        # Add normal distribution overlay
        from scipy import stats
        mu, std = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax5.plot(x, stats.norm.pdf(x, mu, std) * len(residuals) * (bins[1] - bins[0]),
                'r-', lw=2, label=f'Normal(μ={mu:.3f}, σ={std:.3f})')
        
        ax5.axvline(x=0, color='green', linestyle='--', lw=2, label='Zero Error')
        ax5.set_xlabel('Residuals (eV)')
        ax5.set_ylabel('Frequency')
        ax5.set_title(f'Residual Distribution - {best_name}')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        skew = stats.skew(residuals)
        kurt = stats.kurtosis(residuals)
        ax5.text(0.02, 0.98, f'Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}', 
                transform=ax5.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_report.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'summary_report.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Summary report saved (PNG & EPS).")
    
    def export_feature_list(self):
        """Export the complete list of features used for training"""
        if not hasattr(self.screener, 'feature_names') or not self.screener.feature_names:
            print("No feature names found. Please ensure model is loaded.")
            return
        
        # Create feature information dictionary
        feature_info = {
            'total_features': len(self.screener.feature_names),
            'all_features': self.screener.feature_names,
            'feature_categories': {
                'atom_features': [f for f in self.screener.feature_names 
                                if f in self.screener.atom_features],
                'substrate_features': [f for f in self.screener.feature_names 
                                     if f in self.screener.substrate_features],
                'categorical_features': [f for f in self.screener.feature_names 
                                       if f in self.screener.categorical_features],
                'engineered_features': [f for f in self.screener.feature_names 
                                      if ('_X_' in f or '_div_' in f or 
                                          f in ['oxygen_affinity_score', 'size_compatibility', 
                                                'electronic_mismatch'])]
            }
        }
        
        # Save as JSON
        json_path = self.output_dir / 'model_features.json'
        with open(json_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Create a readable text file
        txt_path = self.output_dir / 'model_features.txt'
        with open(txt_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FEATURES USED FOR SAC MODEL TRAINING\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total number of features: {feature_info['total_features']}\n\n")
            
            f.write("FEATURE CATEGORIES:\n")
            f.write("-"*40 + "\n\n")
            
            # Atom features
            f.write(f"1. Atom Features ({len(feature_info['feature_categories']['atom_features'])} features):\n")
            for feat in sorted(feature_info['feature_categories']['atom_features']):
                f.write(f"   - {feat}\n")
            f.write("\n")
            
            # Substrate features
            f.write(f"2. Substrate Features ({len(feature_info['feature_categories']['substrate_features'])} features):\n")
            for feat in sorted(feature_info['feature_categories']['substrate_features']):
                f.write(f"   - {feat}\n")
            f.write("\n")
            
            # Categorical features
            f.write(f"3. Categorical Features ({len(feature_info['feature_categories']['categorical_features'])} features):\n")
            for feat in sorted(feature_info['feature_categories']['categorical_features']):
                f.write(f"   - {feat}\n")
            f.write("\n")
            
            # Engineered features
            f.write(f"4. Engineered Features ({len(feature_info['feature_categories']['engineered_features'])} features):\n")
            for feat in sorted(feature_info['feature_categories']['engineered_features']):
                f.write(f"   - {feat}\n")
            f.write("\n")
            
            # All features alphabetically
            f.write("ALL FEATURES (alphabetical order):\n")
            f.write("-"*40 + "\n")
            for i, feat in enumerate(sorted(self.screener.feature_names), 1):
                f.write(f"{i:3d}. {feat}\n")
        
        # Create a CSV file for easy import
        csv_path = self.output_dir / 'model_features.csv'
        feature_df = pd.DataFrame({
            'feature_name': self.screener.feature_names,
            'feature_index': range(len(self.screener.feature_names))
        })
        
        # Add feature type
        def get_feature_type(feat):
            if feat in self.screener.atom_features:
                return 'atom'
            elif feat in self.screener.substrate_features:
                return 'substrate'
            elif feat in self.screener.categorical_features:
                return 'categorical'
            elif '_X_' in feat or '_div_' in feat:
                return 'interaction'
            elif feat in ['oxygen_affinity_score', 'size_compatibility', 'electronic_mismatch']:
                return 'chemistry_inspired'
            else:
                return 'other'
        
        feature_df['feature_type'] = feature_df['feature_name'].apply(get_feature_type)
        feature_df = feature_df.sort_values('feature_name')
        feature_df.to_csv(csv_path, index=False)
        
        print(f"\nFeature list exported:")
        print(f"  - JSON format: {json_path}")
        print(f"  - Text format: {txt_path}")
        print(f"  - CSV format: {csv_path}")
        print(f"\nTotal features used: {len(self.screener.feature_names)}")
        
        # Also create a feature importance plot if best model supports it
        if hasattr(self.screener.best_model, 'feature_importances_'):
            self._plot_feature_importance_detailed()
    
    def _plot_feature_importance_detailed(self):
        """Create a detailed feature importance plot for the best model"""
        if not hasattr(self.screener.best_model, 'feature_importances_'):
            return
        
        importances = self.screener.best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.screener.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Create two plots: top 30 and category summary
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
        
        # Plot 1: Top 30 features
        top_30 = feature_importance.head(30)
        y_pos = np.arange(len(top_30))
        
        ax1.barh(y_pos, top_30['importance'].values, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_30['feature'].values)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title(f'Top 30 Most Important Features - {self.screener.best_model_name}')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(top_30['importance'].values):
            ax1.text(v + 0.0001, i, f'{v:.4f}', va='center', fontsize=8)
        
        # Plot 2: Importance by category
        def get_category(feat):
            if feat in self.screener.atom_features:
                return 'Atom Properties'
            elif feat in self.screener.substrate_features:
                return 'Substrate Properties'
            elif feat in self.screener.categorical_features:
                return 'Categorical'
            elif '_X_' in feat:
                return 'Interaction (Product)'
            elif '_div_' in feat:
                return 'Interaction (Ratio)'
            elif feat in ['oxygen_affinity_score', 'size_compatibility', 'electronic_mismatch']:
                return 'Chemistry-Inspired'
            else:
                return 'Other'
        
        feature_importance['category'] = feature_importance['feature'].apply(get_category)
        category_importance = feature_importance.groupby('category')['importance'].agg(['sum', 'mean', 'count'])
        category_importance = category_importance.sort_values('sum', ascending=False)
        
        # Create stacked info
        categories = category_importance.index
        sums = category_importance['sum'].values
        counts = category_importance['count'].values
        
        # Create bars
        bars = ax2.bar(categories, sums, alpha=0.8, color='skyblue', edgecolor='navy')
        
        # Add count labels
        for i, (bar, count, sum_val) in enumerate(zip(bars, counts, sums)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{sum_val:.3f}\n({count} features)', 
                    ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Feature Category')
        ax2.set_ylabel('Total Importance')
        ax2.set_title('Feature Importance by Category')
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance_detailed.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'feature_importance_detailed.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        print("Detailed feature importance plot saved (PNG & EPS).")
    
    def run_complete_analysis(self, csv_path: str):
        """Run the complete visual analysis pipeline"""
        print("="*60)
        print("SAC MODEL COMPARISON VISUAL ANALYSIS")
        print("="*60)
        
        # Load data for evaluation
        self.load_data_for_evaluation(csv_path)
        
        # Get top models from already trained set
        self.get_top_models(n_models=6)
        
        # Create all visualizations
        print("\nGenerating visualizations...")
        
        # Export feature list first
        self.export_feature_list()
        
        self.plot_performance_comparison()
        self.plot_predictions_scatter()
        self.plot_residual_analysis()
        self.plot_feature_importance()
        self.plot_cross_validation_scores()
        self.plot_prediction_distributions()
        self.plot_error_analysis()
        self.create_summary_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nAll plots saved to: {self.output_dir}/")
        print("\nGenerated files:")
        print("Features:")
        print("  - model_features.json - Feature list in JSON format")
        print("  - model_features.txt - Human-readable feature list")
        print("  - model_features.csv - Feature list for data analysis")
        print("  - feature_importance_detailed.png/eps - Detailed importance analysis (if applicable)")
        print("\nPlots (PNG & EPS formats):")
        print("  1. performance_comparison - Bar charts comparing key metrics")
        print("  2. predictions_scatter - Actual vs predicted scatter plots")
        print("  3. residual_analysis - Detailed residual diagnostics")
        print("  4. feature_importance_comparison - Feature importance across models")
        print("  5. cv_scores_distribution - Cross-validation score distributions")
        print("  6. prediction_distributions - Distribution of predictions")
        print("  7. error_analysis - Error analysis by data characteristics")
        print("  8. summary_report - Comprehensive summary report")
        print(f"\nBest Model Selected: {self.screener.best_model_name}")


# Function to run analysis with existing trained screener
def analyze_trained_models(screener, csv_path: str, output_dir: str = "model_comparison_plots"):
    """
    Analyze models from an already trained screener
    
    Args:
        screener: Trained SACCatalystScreener object
        csv_path: Path to the data CSV for evaluation
        output_dir: Directory to save plots
    """
    visualizer = SACModelVisualizer(screener=screener, output_dir=output_dir)
    visualizer.run_complete_analysis(csv_path)
    return visualizer


# Main execution
if __name__ == "__main__":
    import sys
    import os
    
    # Handle command line arguments more robustly
    csv_path = None
    
    # Check for valid CSV file argument (skip notebook-specific args like '-f')
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.endswith('.csv') and not arg.startswith('-'):
                csv_path = arg
                break
    
    # If no valid CSV argument, try to find the file
    if csv_path is None:
        if os.path.exists('sac_features_consolidated_fixed.csv'):
            csv_path = 'sac_features_consolidated_fixed.csv'
            print(f"Using fixed data file: {csv_path}")
        elif os.path.exists('sac_features_consolidated.csv'):
            csv_path = 'sac_features_consolidated.csv'
            print(f"Using data file: {csv_path}")
        else:
            print("ERROR: No CSV file found!")
            print("Please provide a CSV file path or ensure 'sac_features_consolidated.csv' exists")
            print("\nUsage:")
            print("  python sac_model_comparison.py <path_to_csv>")
            print("\nOr in Python/Jupyter:")
            print("  # Option 1: Use with already trained screener")
            print("  from sac_complete_system import train_sac_model")
            print("  screener = train_sac_model('your_data.csv')")
            print("  from sac_model_comparison import analyze_trained_models")
            print("  analyze_trained_models(screener, 'your_data.csv')")
            print("\n  # Option 2: Load saved model")
            print("  visualizer = SACModelVisualizer()")
            print("  visualizer.run_complete_analysis('your_data.csv')")
            sys.exit(1)
    
    # Try to load existing model first
    try:
        visualizer = SACModelVisualizer()
        visualizer.run_complete_analysis(csv_path)
    except FileNotFoundError:
        print("No saved model found. Please train the model first using sac_complete_workflow.py")
