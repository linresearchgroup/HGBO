#!/usr/bin/env python3
"""
PFAS Content Prediction using Gaussian Process Regression

This module provides a comprehensive pipeline for predicting PFAS content in soil samples
using Gaussian Process Regression with enhanced kernels, SHAP interpretability analysis,
and Partial Dependence Plot (PDP) visualization.

Key Features:
- Enhanced Gaussian Process regression with custom kernels
- Model interpretability using SHAP analysis  
- Partial dependence analysis (1D and 2D)
- Comprehensive visualization and evaluation tools

Author: PhD Research Project
Date: 2024
Purpose: Reproducible research for PFAS content prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path

# Avoid convergence warning in learning curve
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.*')
warnings.filterwarnings('ignore', module='sklearn.gaussian_process.*')
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except ImportError:
    pass

warnings.filterwarnings('ignore', message='.*optimal value found.*bound.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*SHAP not available.*')

# Core ML imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import (
    train_test_split, learning_curve, ShuffleSplit
)

# Optional imports with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. SHAP analysis will be skipped.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x

try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False


class PFASGaussianProcessModel:
    """
    A comprehensive Gaussian Process model for PFAS content prediction.
    
    This class encapsulates the complete workflow from data loading to model
    training, evaluation, and interpretability analysis.
    """
    
    def __init__(self, data_file: str = "PFAS soil Database.xlsx", random_state: int = 42):
        """
        Initialize the PFAS Gaussian Process model.
        
        Parameters:
        -----------
        data_file : str
            Path to the Excel data file containing PFAS soil data
        random_state : int
            Random state for reproducibility
        """
        self.data_file = data_file
        self.random_state = random_state
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        self.n_data = None
        
        # Feature names used in the model
        self.feature_names = [
            'Material Name', 'Weight_ratio', 'Initial total mass', 
            'Init. Res.(Ω)', 'Init. Pulse Voltage'
        ]
        
    def load_and_preprocess_data(self) -> None:
        """
        Load and preprocess the PFAS dataset from Excel file.
        """
        print("Loading and preprocessing PFAS soil data...")
        
        # Load main data from Excel
        data = pd.read_excel(self.data_file, nrows=52)
        data = data[['Init. Pulse Voltage', 'PFAS Content', 'F ion Content']]
        
        # Load additional data from Processing module if available
        try:
            from Processing import cdata
            new_data = cdata.copy().reset_index(drop=True)
            n_data = pd.concat([new_data, data.reset_index(drop=True)], axis=1)
        except ImportError:
            print("Warning: Processing module not found. Using Excel data only.")
            # Create a basic structure if Processing module is not available
            n_data = data.copy()
            # Add dummy columns that might be expected
            for col in self.feature_names:
                if col not in n_data.columns:
                    n_data[col] = 0  # This will need to be handled appropriately
        
        # Load additional records if available
        try:
            data1 = pd.read_excel(self.data_file, nrows=51)
            if len(data1) > 61:
                updata = data1.loc[58:61, [
                    'Material Name', 'Weight_ratio', 'Initial total mass', 
                    'Init. Res.(Ω)', 'Init. Pulse Voltage', 'PFAS Content', 'F ion Content'
                ]]
                n_data = pd.concat([n_data, updata], ignore_index=True)
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not load additional data: {e}")
        
        # Remove specific outlier
        n_data = n_data[n_data['PFAS Content'] != 0.019]
        
        # Log-transform PFAS Content for better modeling
        n_data['Log_PFAS_Content'] = np.log(n_data['PFAS Content'])
        
        # Store data
        self.n_data = n_data
        self.X = n_data[self.feature_names]
        self.y = n_data['Log_PFAS_Content']
        
        print(f"Data loaded successfully. Shape: {self.X.shape}")
        print(f"Features: {list(self.X.columns)}")
        
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline for feature transformation.
        
        Returns:
        --------
        ColumnTransformer
            Sklearn preprocessing pipeline
        """
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), [
                    'Initial total mass', 'Init. Res.(Ω)', 'Init. Pulse Voltage'
                ]),
                ('cat', OneHotEncoder(), ['Material Name', 'Weight_ratio'])
            ]
        )
    
    def create_enhanced_kernel(self, n_features: int):
        """
        Create an enhanced kernel for Gaussian Process regression.
        
        This kernel combines multiple Matern kernels with different length scales,
        a Rational Quadratic kernel for flexibility, and a White kernel for noise.
        
        Parameters:
        -----------
        n_features : int
            Number of features after preprocessing
            
        Returns:
        --------
        Enhanced kernel object for GP regression
        """
        # Define main Matern kernels with different characteristics
        main_kernel1 = Matern(
            length_scale=1 * np.ones(n_features),
            length_scale_bounds=(np.sqrt(13e-1), np.sqrt(5e5)), 
            nu=0.5
        )
        main_kernel2 = Matern(
            length_scale=2 * np.ones(n_features),
            length_scale_bounds=(np.sqrt(1.65e-1), np.sqrt(1e3)), 
            nu=0.5
        )
        
        # Create composite kernels with constant scaling
        kernel1 = (
            ConstantKernel(
                constant_value=1.0, 
                constant_value_bounds=(np.sqrt(1e-5), np.sqrt(1000))
            ) * main_kernel1 +
            ConstantKernel(
                constant_value=1.0, 
                constant_value_bounds=(np.exp(-100), np.exp(0))
            )
        )
        
        kernel2 = (
            ConstantKernel(
                constant_value=1.0, 
                constant_value_bounds=(np.sqrt(1e-5), np.sqrt(100))
            ) * main_kernel2 +
            ConstantKernel(
                constant_value=1.0, 
                constant_value_bounds=(np.exp(-10), np.exp(0))
            )
        )
        
        # Enhanced kernel combining multiple components
        enhanced_kernel = (
            kernel1 + 
            RationalQuadratic(length_scale=1.0, alpha=1.0) +
            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        )
        
        return enhanced_kernel
    
    def train_model(self, test_size: float = 0.2) -> None:
        """
        Train the Gaussian Process model with enhanced kernel.
        
        Parameters:
        -----------
        test_size : float
            Fraction of data to use for testing (default: 0.2)
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")
        
        print("Training Gaussian Process model with enhanced kernel...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        
        # Create and fit preprocessor to get feature dimensions
        self.preprocessor = self.create_preprocessor()
        X_transformed = self.preprocessor.fit_transform(self.X_train)
        n_transformed_var = X_transformed.shape[1]
        
        # Create enhanced kernel
        enhanced_kernel = self.create_enhanced_kernel(n_transformed_var)
        
        # Create the complete pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('gp', GaussianProcessRegressor(
                kernel=enhanced_kernel,
                n_restarts_optimizer=10,
                random_state=self.random_state
            ))
        ])
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed successfully.")
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the trained model performance.
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call train_model().")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': mean_squared_error(self.y_train, y_train_pred),
            'test_rmse': mean_squared_error(self.y_test, y_test_pred),
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred)
        }
        
        # Print results
        print("\\nModel Performance Metrics:")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"Testing RMSE: {metrics['test_rmse']:.4f}")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Testing R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def plot_learning_curve(self, train_sizes: np.ndarray = None, cv: int = 5) -> None:
        """
        Plot learning curve for the model to assess training dynamics.
        
        Parameters:
        -----------
        train_sizes : np.ndarray, optional
            Training set sizes to use for learning curve
        cv : int
            Number of cross-validation folds
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call train_model().")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.001, 1.0, 30)
        
        print("Generating learning curve...")
        
        # Preprocess data for learning curve
        X_transformed = self.model.named_steps['preprocessor'].fit_transform(self.X)
        
        # Use ShuffleSplit for better performance estimation
        cv_splitter = ShuffleSplit(n_splits=cv, test_size=0.2, random_state=self.random_state)
        
        # Calculate learning curve
        train_sizes_abs, train_scores, test_scores = learning_curve(
            self.model.named_steps['gp'], X_transformed, self.y,
            cv=cv_splitter,
            train_sizes=train_sizes,
            scoring='neg_mean_squared_error',
            error_score='raise'
        )
        
        # Convert to positive MSE
        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', label='Training MSE', linewidth=2)
        plt.plot(train_sizes_abs, test_scores_mean, 'o-', label='Validation MSE', linewidth=2)
        plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        plt.title('Learning Curve (MSE) with Enhanced Kernel', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self) -> None:
        """
        Plot true vs predicted values for both training and test sets.
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call train_model().")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = mean_squared_error(self.y_train, y_train_pred)
        test_rmse = mean_squared_error(self.y_test, y_test_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_train, y_train_pred, color="blue", alpha=0.7,
                   label=f"Training Data\\nR²: {train_r2:.3f}, MSE: {train_rmse:.3f}")
        plt.scatter(self.y_test, y_test_pred, color="green", alpha=0.7,
                   label=f"Testing Data\\nR²: {test_r2:.3f}, MSE: {test_rmse:.3f}")
        
        # Add ideal fit line
        min_val = min(self.y_train.min(), self.y_test.min())
        max_val = max(self.y_train.max(), self.y_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                color="red", linestyle="--", linewidth=2, label="Ideal Fit")
        
        plt.xlabel("True Log_PFAS_Content", fontsize=12, fontweight='bold')
        plt.ylabel("Predicted Log_PFAS_Content", fontsize=12, fontweight='bold')
        plt.title("True vs Predicted Log_PFAS_Content", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_single_sample(self, sample_data: Dict[str, Union[str, float]]) -> Tuple[float, float]:
        """
        Make prediction for a single sample with uncertainty estimation.
        
        Parameters:
        -----------
        sample_data : dict
            Dictionary containing feature values for prediction
            
        Returns:
        --------
        tuple
            (predicted_pfas_content, prediction_std)
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call train_model().")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([sample_data])
        
        # Make prediction with uncertainty
        mu, sigma = self.model.predict(input_df, return_std=True)
        
        # Convert back from log scale
        predicted_pfas = np.exp(mu[0])
        
        return predicted_pfas, sigma[0]
    
    def save_model(self, filepath: str = "gp_pfas_model.joblib") -> None:
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained first. Call train_model().")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'data_file': self.data_file
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data.get('random_state', 42)
        self.data_file = model_data.get('data_file', "PFAS soil Database.xlsx")
        print(f"Model loaded from {filepath}")


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analyzer for model interpretability.
    
    This class provides comprehensive SHAP analysis including feature importance
    ranking and various visualization options.
    """
    
    def __init__(self, model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Initialize SHAP analyzer.
        
        Parameters:
        -----------
        model : Pipeline
            Trained GP model pipeline
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for interpretability analysis. Install with: pip install shap")
        
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.merged_feature_names = None
        self.shap_values_merged = None
    
    def create_explainer(self, background_sample_size: int = None) -> None:
        """
        Create SHAP explainer for the model.
        
        Parameters:
        -----------
        background_sample_size : int, optional
            Number of background samples to use for explanation
        """
        print("Creating SHAP explainer...")
        
        # Prepare background data
        background = self.X_train.copy()
        if background_sample_size and background_sample_size < len(background):
            background = background.sample(n=background_sample_size, random_state=42)
        
        background_transformed = self.model.named_steps['preprocessor'].transform(background)
        
        # Define prediction function
        def model_predict(X):
            return self.model.named_steps['gp'].predict(X)
        
        # Create explainer
        self.explainer = shap.KernelExplainer(model_predict, background_transformed)
        print("SHAP explainer created successfully.")
    
    def calculate_shap_values(self, n_samples: int = 200) -> None:
        """
        Calculate SHAP values for test data.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples for SHAP calculation
        """
        if self.explainer is None:
            self.create_explainer()
        
        print("Calculating SHAP values (this may take a few minutes)...")
        X_explain_transformed = self.model.named_steps['preprocessor'].transform(self.X_test)
        
        self.shap_values = self.explainer.shap_values(X_explain_transformed, nsamples=n_samples)
        
        # Process feature names and merge related features
        self._process_feature_names()
        print("SHAP values calculated successfully.")
    
    def _process_feature_names(self) -> None:
        """
        Process and merge feature names for better interpretability.
        """
        # Get raw feature names from preprocessor
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        
        # Clean and categorize feature names
        raw_names = []
        for name in feature_names:
            # Remove prefixes
            if name.startswith('num__'):
                name = name[5:]
            elif name.startswith('cat__'):
                name = name[5:]
            
            # Categorize features
            if 'Init. Pulse Voltage' in name:
                raw_names.append('Voltage')
            elif 'Initial total mass' in name:
                raw_names.append('Mass')
            elif 'Init. Res' in name:
                raw_names.append('Resistance')
            elif 'Material Name_' in name:
                material = name.replace('Material Name_', '')
                if material.startswith('c-Soil(PFOA):'):
                    material = material.replace('c-Soil(PFOA):', '')
                if material in ['biochar', 'metcoke', 'carbon black']:
                    raw_names.append('Additive Materials')
                else:
                    raw_names.append(material)
            elif 'Weight_ratio_' in name:
                raw_names.append('Additive Ratio')
            else:
                raw_names.append(name)
        
        # Merge features with same names
        merged_name_to_indices = {}
        for idx, name in enumerate(raw_names):
            merged_name_to_indices.setdefault(name, []).append(idx)
        
        self.merged_feature_names = list(merged_name_to_indices.keys())
        
        # Merge SHAP values for related features
        X_explain_transformed = self.model.named_steps['preprocessor'].transform(self.X_test)
        self.shap_values_merged = np.zeros((self.shap_values.shape[0], len(self.merged_feature_names)))
        X_explain_merged = np.zeros((X_explain_transformed.shape[0], len(self.merged_feature_names)))
        
        for i, name in enumerate(self.merged_feature_names):
            idxs = merged_name_to_indices[name]
            self.shap_values_merged[:, i] = self.shap_values[:, idxs].sum(axis=1)
            X_explain_merged[:, i] = X_explain_transformed[:, idxs].sum(axis=1)
    
    def plot_shap_summary(self, plot_type: str = "dot") -> None:
        """
        Plot SHAP summary visualization.
        
        Parameters:
        -----------
        plot_type : str
            Type of plot ("dot" or "bar")
        """
        if self.shap_values_merged is None:
            self.calculate_shap_values()
        
        # Set up plot styling
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'lines.linewidth': 2.0,
            'font.family': 'Times New Roman',
            'font.weight': 'bold',
            'axes.linewidth': 2.0
        })
        
        # Create the plot
        fig = plt.figure(figsize=(8, 4), dpi=600)
        ax = fig.add_subplot(111)
        
        X_explain_transformed = self.model.named_steps['preprocessor'].transform(self.X_test)
        X_explain_merged = np.zeros((X_explain_transformed.shape[0], len(self.merged_feature_names)))
        
        # Recreate merged data for plotting
        merged_name_to_indices = {}
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        raw_names = []
        for name in feature_names:
            if name.startswith('num__'):
                name = name[5:]
            elif name.startswith('cat__'):
                name = name[5:]
            if 'Init. Pulse Voltage' in name:
                raw_names.append('Voltage')
            elif 'Initial total mass' in name:
                raw_names.append('Mass')
            elif 'Init. Res' in name:
                raw_names.append('Resistance')
            elif 'Material Name_' in name:
                material = name.replace('Material Name_', '')
                if material.startswith('c-Soil(PFOA):'):
                    material = material.replace('c-Soil(PFOA):', '')
                if material in ['biochar', 'metcoke', 'carbon black']:
                    raw_names.append('Additive Materials')
                else:
                    raw_names.append(material)
            elif 'Weight_ratio_' in name:
                raw_names.append('Additive Ratio')
            else:
                raw_names.append(name)
        
        for idx, name in enumerate(raw_names):
            merged_name_to_indices.setdefault(name, []).append(idx)
        
        for i, name in enumerate(self.merged_feature_names):
            idxs = merged_name_to_indices[name]
            X_explain_merged[:, i] = X_explain_transformed[:, idxs].sum(axis=1)
        
        if plot_type == "bar":
            shap.summary_plot(self.shap_values_merged, X_explain_merged, 
                            feature_names=self.merged_feature_names, plot_type="bar", show=False)
            ax.set_xlabel("mean(|SHAP value|)", fontsize=16, fontweight='bold')
        else:
            shap.summary_plot(self.shap_values_merged, X_explain_merged, 
                            feature_names=self.merged_feature_names, show=False)
            plt.xlabel("SHAP value", fontsize=16, fontweight='bold')
        
        ax = plt.gca()
        ax.set_ylabel(ax.get_ylabel(), fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16, length=10, width=3)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        ax.tick_params(width=2, colors='black')
        plt.grid(False)
        plt.tight_layout()
        fig.set_size_inches(8, 4)
        plt.show()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance ranking based on SHAP values.
        
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe sorted by importance
        """
        if self.shap_values_merged is None:
            self.calculate_shap_values()
        
        importance_df = pd.DataFrame({
            'Feature Names': self.merged_feature_names,
            'SHAP Importance': np.abs(self.shap_values_merged).mean(0)
        })
        
        return importance_df.sort_values('SHAP Importance', ascending=False)


class PDPAnalyzer:
    """
    Partial Dependence Plot (PDP) analyzer for understanding feature effects.
    
    This class provides 1D and 2D partial dependence analysis with ICE curves
    for comprehensive feature effect visualization.
    """
    
    def __init__(self, model: Pipeline, X: pd.DataFrame):
        """
        Initialize PDP analyzer.
        
        Parameters:
        -----------
        model : Pipeline
            Trained GP model pipeline
        X : pd.DataFrame
            Feature data for analysis
        """
        self.model = model
        self.X = X
    
    def compute_ice_curves(self, feature: str, grid_resolution: int = 50, 
                          sample_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Individual Conditional Expectation (ICE) curves and PDP values.
        
        Parameters:
        -----------
        feature : str
            Feature name for PDP calculation
        grid_resolution : int
            Number of grid points for the feature range
        sample_size : int, optional
            Number of samples for ICE curve calculation
            
        Returns:
        --------
        tuple
            (ice_curves, grid, pd_values)
        """
        # Get feature range
        feature_min, feature_max = self.X[feature].min(), self.X[feature].max()
        grid = np.linspace(feature_min, feature_max, grid_resolution)
        
        # Sample data if specified
        if sample_size is not None and sample_size < len(self.X):
            sample_indices = np.random.choice(len(self.X), sample_size, replace=False)
            X_sampled = self.X.iloc[sample_indices]
        else:
            X_sampled = self.X
        
        # Calculate ICE curves
        n_samples = len(X_sampled)
        ice_curves = np.zeros((n_samples, grid_resolution))
        
        for i, val in enumerate(grid):
            X_mod = X_sampled.copy()
            X_mod[feature] = val
            ice_curves[:, i] = self.model.predict(X_mod)
        
        # Calculate PDP (average of ICE curves)
        pd_values = np.mean(ice_curves, axis=0)
        
        return ice_curves, grid, pd_values
    
    def create_pdp_plot(self, feature: str, feature_label: str = None, 
                       sample_ice_size: int = 30, smoothing: float = 0.5,
                       save_path: str = None) -> None:
        """
        Create and display 1D PDP plot with ICE curves.
        
        Parameters:
        -----------
        feature : str
            Feature name for analysis
        feature_label : str, optional
            Label for the feature in the plot
        sample_ice_size : int
            Number of ICE curves to display
        smoothing : float
            Gaussian smoothing standard deviation
        save_path : str, optional
            Path to save the plot
        """
        if feature_label is None:
            feature_label = feature
        
        print(f"Calculating PDP and ICE for {feature}...")
        ice_curves, grid, pd_values = self.compute_ice_curves(
            feature, sample_size=sample_ice_size
        )
        
        # Create plot with styling
        plt.figure(figsize=(12, 10), dpi=600)
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # Apply smoothing if available
        if SCIPY_AVAILABLE and smoothing > 0:
            pd_smooth = gaussian_filter1d(pd_values, sigma=smoothing)
        else:
            pd_smooth = pd_values
        
        # Center the curves
        reference_point = np.mean(pd_smooth)
        pd_centered = pd_smooth - reference_point
        
        # Plot ICE curves (centered)
        if ice_curves is not None and len(ice_curves) > 0:
            ice_centered = ice_curves - np.mean(ice_curves, axis=1).reshape(-1, 1)
            for j in range(min(len(ice_centered), 40)):
                plt.plot(grid, ice_centered[j], color='steelblue', alpha=1, linewidth=1)
        
        # Plot average PDP curve
        plt.plot(grid, pd_centered, color='darkred', linestyle='--', 
                linewidth=2.5, label='average')
        
        plt.legend(loc='upper left', frameon=False, fontsize=35)
        
        # Add data distribution markers
        plt.plot(self.X[feature], np.ones_like(self.X[feature]) * plt.ylim()[0], '|', 
                color='black', alpha=1, markersize=10)
        
        plt.xlabel(feature_label, fontsize=35, fontweight='bold', fontname='Times New Roman')
        plt.ylabel('Partial dependence', fontsize=35, fontweight='bold', fontname='Times New Roman')
        
        # Remove grid
        plt.grid(False)
        
        # Special handling for voltage
        if feature == 'Init. Pulse Voltage' or feature_label == 'Voltage (V)':
            y_min, y_max = plt.ylim()
            plt.ylim(y_min - 0.5, y_max + 0.5)
        
        # Style the plot
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_color('black')
            spine.set_visible(True)
        
        ax.tick_params(axis='both', which='major', labelsize=30, length=10, width=3)
        
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        else:
            # Auto-generate filename
            feature_name = feature.replace(' ', '_').replace('.', '')
            plt.savefig(f'pdp_1d_{feature_name}.png', dpi=600, bbox_inches='tight')
        
        plt.show()
    
    def create_multiple_pdp_plots(self, features: List[str], feature_labels: List[str] = None,
                                 sample_ice_size: int = 30, smoothing: float = 0.5) -> None:
        """
        Create multiple 1D PDP plots for different features.
        
        Parameters:
        -----------
        features : list
            List of feature names for PDP calculation
        feature_labels : list, optional
            List of feature labels for display
        sample_ice_size : int
            Number of ICE curves to display
        smoothing : float
            Gaussian smoothing standard deviation
        """
        if feature_labels is None:
            feature_labels = features
        
        for feature, label in zip(features, feature_labels):
            self.create_pdp_plot(feature, label, sample_ice_size, smoothing)
    
    def compute_2d_pdp(self, feature1: str, feature2: str, 
                      grid_resolution: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D partial dependence for two features.
        
        Parameters:
        -----------
        feature1, feature2 : str
            Names of two features for PDP calculation
        grid_resolution : int
            Number of grid points in each dimension
            
        Returns:
        --------
        tuple
            (x_grid, y_grid, pdp_values)
        """
        # Get feature ranges
        f1_min, f1_max = self.X[feature1].min(), self.X[feature1].max()
        f2_min, f2_max = self.X[feature2].min(), self.X[feature2].max()
        
        x_grid = np.linspace(f1_min, f1_max, grid_resolution)
        y_grid = np.linspace(f2_min, f2_max, grid_resolution)
        
        pdp_values = np.zeros((grid_resolution, grid_resolution))
        
        # Calculate PDP with progress tracking
        iterator = enumerate(x_grid)
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(x_grid), 
                          desc=f"Calculating 2D PDP for {feature1} vs {feature2}")
        
        for i, x_val in iterator:
            for j, y_val in enumerate(y_grid):
                X_mod = self.X.copy()
                X_mod[feature1] = x_val
                X_mod[feature2] = y_val
                pdp_values[j, i] = np.mean(self.model.predict(X_mod))
        
        # Center PDP values
        pdp_values = pdp_values - np.mean(pdp_values)
        
        return x_grid, y_grid, pdp_values
    
    def plot_2d_pdp(self, x_grid: np.ndarray, y_grid: np.ndarray, pdp_values: np.ndarray,
                   feature1_label: str, feature2_label: str, 
                   cmap: str = 'Spectral', save_path: str = None) -> None:
        """
        Create and display 2D PDP plot.
        
        Parameters:
        -----------
        x_grid, y_grid : np.ndarray
            Grid points for the two features
        pdp_values : np.ndarray
            2D PDP values
        feature1_label, feature2_label : str
            Labels for the features in the plot
        cmap : str
            Colormap for the plot
        save_path : str, optional
            Path to save the plot
        """
        # Create plot with styling
        fig = plt.figure(figsize=(12, 10), dpi=600)
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # Determine symmetric color range
        abs_max = max(abs(np.min(pdp_values)), abs(np.max(pdp_values)))
        vmin, vmax = -abs_max, abs_max
        
        # Create main plot
        mesh = plt.pcolormesh(x_grid, y_grid, pdp_values, cmap=cmap, 
                             vmin=vmin, vmax=vmax, shading='auto')
        
        # Add contour lines
        CS = plt.contour(x_grid, y_grid, pdp_values, 10, colors='black', 
                        linestyles='dashed', linewidths=0.5)
        
        # Style the plot
        plt.xlabel(feature1_label, fontsize=32, fontweight='bold', fontname='Times New Roman')
        plt.ylabel(feature2_label, fontsize=32, fontweight='bold', fontname='Times New Roman')
        plt.grid(False)
        
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_color('black')
            spine.set_visible(True)
        
        ax.tick_params(axis='both', which='major', labelsize=30, width=3, 
                      colors='black', length=10)
        
        # Add colorbar
        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', 
                           pad=0.02, fraction=0.05, aspect=40)
        cbar.set_label('Partial dependence', fontsize=28, fontweight='bold', 
                      fontname='Times New Roman', labelpad=20)
        cbar.ax.tick_params(labelsize=25, width=3, length=6)
        cbar.outline.set_linewidth(2)
        cbar.outline.set_edgecolor('black')
        
        # Style colorbar labels
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontname('Times New Roman')
        
        # Style axis labels
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        plt.tight_layout(rect=[0, 0, 0.96, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            # Auto-generate filename
            file_name = f'pdp_2d_{feature1_label.replace(" ", "_").replace("(", "").replace(")", "")}_{feature2_label.replace(" ", "_").replace("(", "").replace(")", "")}.png'
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_multiple_2d_pdp_plots(self, feature_pairs: List[Tuple[str, str]], 
                                    feature_labels_dict: Dict[str, str],
                                    grid_resolution: int = 30) -> None:
        """
        Create multiple 2D PDP plots for different feature pairs.
        
        Parameters:
        -----------
        feature_pairs : list of tuples
            List of feature pairs for PDP calculation
        feature_labels_dict : dict
            Mapping from feature names to display labels
        grid_resolution : int
            Grid resolution for each dimension
        """
        for feature1, feature2 in feature_pairs:
            print(f"\\nCalculating 2D PDP for {feature1} vs {feature2}...")
            feature1_label = feature_labels_dict.get(feature1, feature1)
            feature2_label = feature_labels_dict.get(feature2, feature2)
            
            # Calculate 2D PDP
            x_grid, y_grid, pdp_values = self.compute_2d_pdp(feature1, feature2, grid_resolution)
            
            # Plot and save 2D PDP
            self.plot_2d_pdp(x_grid, y_grid, pdp_values, feature1_label, feature2_label)


def main():
    """
    Main function demonstrating the complete PFAS prediction pipeline.
    """
    print("=== PFAS Content Prediction using Gaussian Process Regression ===\\n")
    
    try:
        # Initialize and train model
        model = PFASGaussianProcessModel(random_state=42)
        
        # Load and preprocess data
        model.load_and_preprocess_data()
        
        # Train model
        model.train_model()
        
        # Evaluate model
        metrics = model.evaluate_model()
        
        # Visualize results
        model.plot_predictions()
        model.plot_learning_curve()
        
        # Save model
        model.save_model("gp_P.joblib")
        
        # SHAP Analysis (if available)
        if SHAP_AVAILABLE:
            print("\\n=== Performing SHAP Analysis ===")
            shap_analyzer = SHAPAnalyzer(model.model, model.X_train, model.X_test)
            shap_analyzer.calculate_shap_values(n_samples=200)
            
            # Plot SHAP summary
            shap_analyzer.plot_shap_summary("dot")
            shap_analyzer.plot_shap_summary("bar")
            
            # Get feature importance
            importance_df = shap_analyzer.get_feature_importance()
            print("\\nFeature Importance Ranking:")
            print(importance_df)
        
        # PDP Analysis
        print("\\n=== Performing Partial Dependence Analysis ===")
        pdp_analyzer = PDPAnalyzer(model.model, model.X)
        
        # 1D PDP for key features
        key_features = ['Init. Pulse Voltage', 'Initial total mass', 'Init. Res.(Ω)']
        feature_labels = ['Voltage (V)', 'Mass (mg)', 'Resistance (Ω)']
        
        pdp_analyzer.create_multiple_pdp_plots(key_features, feature_labels, 
                                              sample_ice_size=30, smoothing=0.5)
        
        # 2D PDP for feature pairs
        feature_pairs = [
            ('Init. Pulse Voltage', 'Initial total mass'),
            ('Init. Pulse Voltage', 'Init. Res.(Ω)'),
            ('Initial total mass', 'Init. Res.(Ω)')
        ]
        
        feature_labels_dict = {
            'Init. Pulse Voltage': 'Voltage (V)',
            'Initial total mass': 'Mass (mg)',
            'Init. Res.(Ω)': 'Resistance (Ω)'
        }
        
        pdp_analyzer.create_multiple_2d_pdp_plots(feature_pairs, feature_labels_dict, 
                                                 grid_resolution=25)
        
        print("\\n=== Analysis Complete ===")
        print("All plots and model files have been saved to the current directory.")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        print("Please check your data file and dependencies.")
        raise


if __name__ == "__main__":
    main()