"""
XGBoost Regression Constraint Model for PFAS Resistance Prediction
=================================================================

This module implements an XGBoost regression model as a constraint model
for predicting initial resistance values in PFAS soil treatment systems.

Author: Generated from XG_R.ipynb
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class XGBoostConstraintModel:
    """
    XGBoost-based constraint model for PFAS resistance prediction.
    
    This class implements a complete machine learning pipeline for predicting
    initial resistance values based on material properties and experimental conditions.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the XGBoost constraint model.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.categorical_features = ['Material Name', 'Weight_ratio']
        self.numerical_features = ['Initial total mass']
        
        # Model performance metrics
        self.train_metrics = {}
        self.test_metrics = {}
        
        # Set plotting style
        plt.rcParams['font.family'] = 'Times New Roman'
        
    def load_and_prepare_data(self, excel_file="PFAS soil Database.xlsx"):
        """
        Load and prepare data for training.
        
        Args:
            excel_file (str): Path to the Excel file containing PFAS data
            
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        try:
            # Import processed data from Processing module
            from Processing import cdata
            
            # Load additional data from Excel file
            data1 = pd.read_excel(excel_file, nrows=52)
            updata = data1.loc[52:52, ['Material Name', 'Weight_ratio', 'Initial total mass', 
                                      'Init. Res.(Ω)', 'Init. Pulse Voltage', 'PFAS Content', 'F ion Content']]
            
            # Combine datasets
            cdata_combined = pd.concat([cdata, updata], ignore_index=True)
            
            # Prepare features and target
            X = cdata_combined.drop(columns=['Init. Res.(Ω)'])
            y = cdata_combined['Init. Res.(Ω)']
            
            print(f"Data loaded successfully. Shape: {X.shape}")
            print(f"Target variable range: {y.min():.2f} - {y.max():.2f}")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def setup_preprocessor(self):
        """
        Set up the data preprocessing pipeline.
        
        Returns:
            ColumnTransformer: Configured preprocessor
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features),
                ('num', StandardScaler(), self.numerical_features)
            ]
        )
        return self.preprocessor
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target vector
            test_size (float): Proportion of data for testing
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
    
    def create_base_model(self):
        """
        Create the base XGBoost model with default parameters.
        
        Returns:
            Pipeline: XGBoost pipeline with preprocessing
        """
        # Define base XGBoost model
        xgb_model = XGBRegressor(
            objective='reg:squarederror',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=self.random_state
        )
        
        # Create pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', xgb_model)
        ])
        
        return self.model
    
    def train_base_model(self):
        """
        Train the base model and evaluate performance.
        
        Returns:
            dict: Training metrics
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_base_model() first.")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"Base Model - Mean Squared Error: {mse:.4f}")
        print(f"Base Model - R² Score: {r2:.4f}")
        
        return {'mse': mse, 'r2': r2}
    
    def hyperparameter_optimization(self, cv_folds=5, n_jobs=-1, verbose=0):
        """
        Perform hyperparameter optimization using GridSearchCV.
        
        Args:
            cv_folds (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
            verbose (int): Verbosity level
            
        Returns:
            dict: Best parameters found
        """
        print("Starting hyperparameter optimization...")
        
        # Create model for optimization
        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', xgb_model)
        ])
        
        # Define parameter grid
        param_grid = {
            'regressor__max_depth': [3, 6, 9],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__n_estimators': [50, 100, 200],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv_folds, 
            scoring='neg_mean_squared_error', 
            verbose=verbose, 
            n_jobs=n_jobs
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        # Store best model
        self.best_model = grid_search.best_estimator_
        
        print("Hyperparameter optimization completed.")
        print(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def evaluate_model(self, model=None, use_filtered_data=False, resistance_threshold=5):
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            model: Model to evaluate (uses best_model if None)
            use_filtered_data (bool): Whether to filter data by resistance threshold
            resistance_threshold (float): Threshold for filtering data
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        if model is None:
            model = self.best_model
        
        if model is None:
            raise ValueError("No model available for evaluation.")
        
        # Prepare data based on filtering option
        if use_filtered_data:
            mask = self.y_train <= resistance_threshold
            X_train_eval = self.X_train[mask]
            y_train_eval = self.y_train[mask]
            
            mask_test = self.y_test <= resistance_threshold
            X_test_eval = self.X_test[mask_test]
            y_test_eval = self.y_test[mask_test]
        else:
            X_train_eval = self.X_train
            y_train_eval = self.y_train
            X_test_eval = self.X_test
            y_test_eval = self.y_test
        
        # Make predictions
        y_train_pred = model.predict(X_train_eval)
        y_test_pred = model.predict(X_test_eval)
        
        # Calculate comprehensive metrics
        train_metrics = {
            'mse': mean_squared_error(y_train_eval, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train_eval, y_train_pred)),
            'mae': mean_absolute_error(y_train_eval, y_train_pred),
            'r2': r2_score(y_train_eval, y_train_pred)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test_eval, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_eval, y_test_pred)),
            'mae': mean_absolute_error(y_test_eval, y_test_pred),
            'r2': r2_score(y_test_eval, y_test_pred)
        }
        
        # Store metrics
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        
        # Print results
        print("\n=== Model Evaluation Results ===")
        print(f"Train MSE: {train_metrics['mse']:.4f}, RMSE: {train_metrics['rmse']:.4f}, "
              f"MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"Test MSE: {test_metrics['mse']:.4f}, RMSE: {test_metrics['rmse']:.4f}, "
              f"MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': {
                'y_train_true': y_train_eval,
                'y_train_pred': y_train_pred,
                'y_test_true': y_test_eval,
                'y_test_pred': y_test_pred
            }
        }
    
    def train_and_evaluate_filtered_model(self, resistance_threshold=5):
        """
        Train a new model using only filtered data (resistance <= threshold) and evaluate performance.
        This implements the filtered data analysis from the original notebook.
        
        Args:
            resistance_threshold (float): Threshold for filtering data (default: 5)
            
        Returns:
            dict: Results including filtered model and performance metrics
        """
        print(f"\n=== Training Model with Filtered Data (R <= {resistance_threshold}) ===")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Split data first.")
        
        # Filter training data
        train_mask = self.y_train <= resistance_threshold
        X_train_filtered = self.X_train[train_mask]
        y_train_filtered = self.y_train[train_mask]
        
        # Filter test data
        test_mask = self.y_test <= resistance_threshold
        X_test_filtered = self.X_test[test_mask]
        y_test_filtered = self.y_test[test_mask]
        
        print(f"Original training set size: {len(self.y_train)}")
        print(f"Filtered training set size: {len(y_train_filtered)} ({len(y_train_filtered)/len(self.y_train)*100:.1f}%)")
        print(f"Original test set size: {len(self.y_test)}")
        print(f"Filtered test set size: {len(y_test_filtered)} ({len(y_test_filtered)/len(self.y_test)*100:.1f}%)")
        
        if len(y_train_filtered) == 0:
            print("Warning: No training data after filtering!")
            return None
        
        # Create new model with best parameters if available
        if hasattr(self, 'best_model') and self.best_model is not None:
            # Extract best parameters from the existing model
            best_regressor = self.best_model.named_steps['regressor']
            filtered_model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', XGBRegressor(
                    objective='reg:squarederror',
                    max_depth=best_regressor.max_depth,
                    learning_rate=best_regressor.learning_rate,
                    n_estimators=best_regressor.n_estimators,
                    subsample=getattr(best_regressor, 'subsample', 1.0),
                    colsample_bytree=getattr(best_regressor, 'colsample_bytree', 1.0),
                    random_state=self.random_state
                ))
            ])
        else:
            # Use default parameters
            filtered_model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', XGBRegressor(
                    objective='reg:squarederror',
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    random_state=self.random_state
                ))
            ])
        
        # Train the filtered model
        print("Training model on filtered data...")
        filtered_model.fit(X_train_filtered, y_train_filtered)
        
        # Make predictions
        y_train_pred_filtered = filtered_model.predict(X_train_filtered)
        y_test_pred_filtered = filtered_model.predict(X_test_filtered)
        
        # Calculate comprehensive metrics for filtered data
        train_metrics_filtered = {
            'mse': mean_squared_error(y_train_filtered, y_train_pred_filtered),
            'rmse': np.sqrt(mean_squared_error(y_train_filtered, y_train_pred_filtered)),
            'mae': mean_absolute_error(y_train_filtered, y_train_pred_filtered),
            'r2': r2_score(y_train_filtered, y_train_pred_filtered)
        }
        
        test_metrics_filtered = {
            'mse': mean_squared_error(y_test_filtered, y_test_pred_filtered),
            'rmse': np.sqrt(mean_squared_error(y_test_filtered, y_test_pred_filtered)),
            'mae': mean_absolute_error(y_test_filtered, y_test_pred_filtered),
            'r2': r2_score(y_test_filtered, y_test_pred_filtered)
        }
        
        # Print filtered model results
        print("\n=== Filtered Model Performance (R <= {:.1f}) ===".format(resistance_threshold))
        print(f"Train MSE: {train_metrics_filtered['mse']:.4f}, RMSE: {train_metrics_filtered['rmse']:.4f}, "
              f"MAE: {train_metrics_filtered['mae']:.4f}, R²: {train_metrics_filtered['r2']:.4f}")
        print(f"Test MSE: {test_metrics_filtered['mse']:.4f}, RMSE: {test_metrics_filtered['rmse']:.4f}, "
              f"MAE: {test_metrics_filtered['mae']:.4f}, R²: {test_metrics_filtered['r2']:.4f}")
        
        # Store filtered model for future use
        self.filtered_model = filtered_model
        self.filtered_threshold = resistance_threshold
        
        return {
            'filtered_model': filtered_model,
            'threshold': resistance_threshold,
            'train_metrics': train_metrics_filtered,
            'test_metrics': test_metrics_filtered,
            'data_sizes': {
                'original_train': len(self.y_train),
                'filtered_train': len(y_train_filtered),
                'original_test': len(self.y_test),
                'filtered_test': len(y_test_filtered)
            },
            'predictions': {
                'y_train_true': y_train_filtered,
                'y_train_pred': y_train_pred_filtered,
                'y_test_true': y_test_filtered,
                'y_test_pred': y_test_pred_filtered
            }
        }
    
    def save_model(self, filepath='xgb_constraint_model.joblib'):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No trained model to save.")
        
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='xgb_constraint_model.joblib'):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            self.best_model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def plot_data_distribution(self, figsize=(12, 4.8)):
        """
        Plot distribution of training and testing data.
        
        Args:
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize, dpi=300)
        sns.histplot(self.y_train, color="#84C3B7", kde=True, label="Train", bins=30)
        sns.histplot(self.y_test, color="#9370DB", kde=True, label="Test", bins=30)
        
        plt.xlabel("R(Ω)", fontsize=32, fontweight='bold')
        plt.ylabel("Frequency", fontsize=32, fontweight='bold')
        
        ax = plt.gca()
        xticks = np.arange(2.5, 21, 2.5)
        ax.set_xticks(xticks)
        ax.tick_params(axis='both', which='major', labelsize=25, length=8, width=2)
        
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            
        legend_font = {'weight': 'bold', 'size': 20}
        ax.legend(frameon=False, prop=legend_font)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("black")
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_scatter(self, evaluation_results=None, figsize=(14, 12)):
        """
        Plot scatter plot of actual vs predicted values with detailed analysis.
        
        Args:
            evaluation_results (dict): Results from evaluate_model
            figsize (tuple): Figure size
        """
        if evaluation_results is None:
            evaluation_results = self.evaluate_model()
        
        # Extract data
        predictions = evaluation_results['predictions']
        y_train_true = predictions['y_train_true']
        y_train_pred = predictions['y_train_pred']
        y_test_true = predictions['y_test_true']
        y_test_pred = predictions['y_test_pred']
        
        # Create combined dataframes
        data_train = pd.DataFrame({
            "R": y_train_true, 
            "predicted_R": y_train_pred, 
            "dataset": "Train"
        })
        data_test = pd.DataFrame({
            "R": y_test_true, 
            "predicted_R": y_test_pred, 
            "dataset": "Test"
        })
        data = pd.concat([data_train, data_test], ignore_index=True)
        
        # Set up colors
        palette = {"Train": "#84C3B7", "Test": "#9370DB"}
        
        # Create figure with subplots
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(7, 7, figure=fig, hspace=0.02, wspace=0.02)
        
        # Top histogram
        ax_histx = fig.add_subplot(gs[0, :-1])
        sns.kdeplot(data=data, x="R", hue="dataset", palette=palette, 
                   fill=True, alpha=0.5, linewidth=1.5, ax=ax_histx)
        ax_histx.get_legend().remove()
        ax_histx.spines["top"].set_visible(False)
        ax_histx.spines["right"].set_visible(False)
        ax_histx.spines["left"].set_visible(False)
        ax_histx.spines["bottom"].set_visible(False)
        ax_histx.set_xticks([])
        ax_histx.set_yticks([])
        ax_histx.set_xlabel("")
        ax_histx.set_ylabel("")
        ax_histx.tick_params(left=False, bottom=False)
        
        # Right histogram
        ax_histy = fig.add_subplot(gs[1:, -1])
        sns.kdeplot(data=data, y="predicted_R", hue="dataset", palette=palette, 
                   fill=True, alpha=0.5, linewidth=1.5, ax=ax_histy)
        ax_histy.get_legend().remove()
        ax_histy.spines["top"].set_visible(False)
        ax_histy.spines["right"].set_visible(False)
        ax_histy.spines["left"].set_visible(False)
        ax_histy.spines["bottom"].set_visible(False)
        ax_histy.set_xticks([])
        ax_histy.set_yticks([])
        ax_histy.set_xlabel("")
        ax_histy.set_ylabel("")
        ax_histy.tick_params(left=False, bottom=False)
        
        # Main scatter plot
        ax_main = fig.add_subplot(gs[1:, :-1])
        sns.scatterplot(data=data, x="R", y="predicted_R", hue="dataset", 
                       palette=palette, alpha=0.8, s=200, ax=ax_main)
        
        # Add perfect prediction line
        ax_main.plot([data["R"].min(), data["R"].max()],
                    [data["R"].min(), data["R"].max()],
                    linestyle="--", color="black")
        
        # Inset plot for zoomed view
        ax_inset = inset_axes(ax_main, width="40%", height="40%", 
                             loc="lower right", borderpad=4)
        data_zoomed = data[data["R"] < 5]
        sns.scatterplot(data=data_zoomed, x="R", y="predicted_R", hue="dataset", 
                       palette=palette, alpha=0.7, s=200, ax=ax_inset)
        ax_inset.get_legend().remove()
        ax_inset.plot([0, 5], [0, 5], linestyle="--", color="black")
        ax_inset.set_xlim(0, 5)
        ax_inset.set_ylim(0, 5)
        
        # Format inset
        xticks = np.arange(1, 5, 1)
        yticks = np.arange(1, 5, 1)
        ax_inset.set_xticks(xticks)
        ax_inset.set_yticks(yticks)
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        ax_inset.tick_params(axis='both', which='major', labelsize=30, 
                           width=3, length=7, direction='in')
        
        for spine in ax_inset.spines.values():
            spine.set_linewidth(2)
        for label in ax_inset.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax_inset.get_yticklabels():
            label.set_fontweight('bold')
        
        # Format main plot
        for spine in ax_main.spines.values():
            spine.set_linewidth(4)
            spine.set_color("black")
        
        # Legend and statistics
        handles, labels = ax_main.get_legend_handles_labels()
        new_labels = ["Train", "Test"]
        legend_font = {'weight': 'bold', 'size': 30}
        legend = ax_main.legend(handles[:2], new_labels, loc="upper left", 
                               frameon=False, prop=legend_font)
        
        # Add statistics text
        legend_bbox = legend.get_window_extent().transformed(ax_main.transAxes.inverted())
        train_metrics = self.train_metrics
        test_metrics = self.test_metrics
        
        stats_text = (f"Train MSE: {train_metrics['mse']:.2f}\n"
                     f"Test MSE: {test_metrics['mse']:.2f}\n"
                     f"Train R²: {train_metrics['r2']:.2f}\n"
                     f"Test R²: {test_metrics['r2']:.2f}")
        
        ax_main.text(0.045, legend_bbox.y0 - 0.3, stats_text, 
                    transform=ax_main.transAxes, fontsize=30, fontweight='bold')
        
        # Labels and formatting
        ax_main.set_xlabel("Actual R(Ω)", fontweight='bold', fontsize=35)
        ax_main.set_ylabel("Predicted R(Ω)", fontweight='bold', fontsize=35)
        
        y_max = np.ceil(max(data['predicted_R'].max(), data['R'].max()))
        ax_main.set_yticks(np.arange(0, y_max + 5, 5))
        ax_main.tick_params(axis='both', which='major', labelsize=30, 
                           length=10, width=3, direction='in')
        
        for label in ax_main.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax_main.get_yticklabels():
            label.set_fontweight('bold')
        
        plt.show()
    
    def plot_residuals(self, evaluation_results=None, figsize=(10, 6)):
        """
        Plot residual distribution for model diagnostics.
        
        Args:
            evaluation_results (dict): Results from evaluate_model
            figsize (tuple): Figure size
        """
        if evaluation_results is None:
            evaluation_results = self.evaluate_model()
        
        predictions = evaluation_results['predictions']
        y_test_true = predictions['y_test_true']
        y_test_pred = predictions['y_test_pred']
        
        # Calculate residuals
        residuals = y_test_true - y_test_pred
        
        # Create plot
        plt.figure(figsize=figsize, dpi=300)
        sns.histplot(residuals, kde=True, bins=20, color="#9F6693", label="Residuals")
        plt.axvline(0, color='red', linestyle='--', label="Zero Error")
        
        plt.xlabel("Residual (True - Predicted)", fontsize=25, fontweight='bold')
        plt.ylabel("Count", fontsize=25, fontweight='bold')
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            
        legend_font = {'weight': 'bold', 'size': 20}
        ax.legend(frameon=False, prop=legend_font)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self, excel_file="PFAS soil Database.xlsx", 
                            save_model_path='xgb_constraint_model.joblib'):
        """
        Run the complete machine learning pipeline.
        
        Args:
            excel_file (str): Path to Excel data file
            save_model_path (str): Path to save the trained model
            
        Returns:
            dict: Complete results including metrics and predictions
        """
        print("=== Starting XGBoost Constraint Model Pipeline ===\n")
        
        # Step 1: Load and prepare data
        print("1. Loading and preparing data...")
        X, y = self.load_and_prepare_data(excel_file)
        if X is None or y is None:
            return None
        
        # Step 2: Setup preprocessing
        print("2. Setting up data preprocessing...")
        self.setup_preprocessor()
        
        # Step 3: Split data
        print("3. Splitting data into train/test sets...")
        self.split_data(X, y)
        
        # Step 4: Create and train base model
        print("4. Creating and training base model...")
        self.create_base_model()
        base_metrics = self.train_base_model()
        
        # Step 5: Hyperparameter optimization
        print("5. Performing hyperparameter optimization...")
        best_params = self.hyperparameter_optimization()
        
        # Step 6: Evaluate optimized model
        print("6. Evaluating optimized model...")
        evaluation_results = self.evaluate_model()
        
        # Step 7: Train and evaluate filtered model (similar to ipynb Cell 6)
        print("7. Training and evaluating filtered model...")
        filtered_results = self.train_and_evaluate_filtered_model(resistance_threshold=5)
        
        # Step 8: Save model
        print("8. Saving trained model...")
        self.save_model(save_model_path)
        
        print("\n=== Pipeline Complete ===")
        
        return {
            'base_metrics': base_metrics,
            'best_params': best_params,
            'evaluation_results': evaluation_results,
            'filtered_results': filtered_results,
            'model_path': save_model_path
        }


def main():
    """
    Main function to demonstrate the XGBoost constraint model.
    """
    # Initialize model
    model = XGBoostConstraintModel(random_state=42)
    
    # Run complete pipeline
    results = model.run_complete_pipeline()
    
    if results is not None:
        print("\n=== Generating Visualizations ===")
        
        # Plot data distribution
        print("Plotting data distribution...")
        model.plot_data_distribution()
        
        # Plot prediction scatter plot
        print("Plotting prediction scatter plot...")
        model.plot_prediction_scatter(results['evaluation_results'])
        
        # Plot residuals
        print("Plotting residuals...")
        model.plot_residuals(results['evaluation_results'])
        
        print("\n=== Demonstration Complete ===")
        
        # Display filtered model results if available
        if 'filtered_results' in results and results['filtered_results'] is not None:
            print("\n=== Filtered Model Analysis (R <= 5) ===")
            filtered_res = results['filtered_results']
            print(f"Data reduction: {filtered_res['data_sizes']['original_train']} → {filtered_res['data_sizes']['filtered_train']} samples")
            # Compare with full model performance
            full_metrics = results['evaluation_results']['test']
            filtered_metrics = filtered_res['test_metrics']
            print(f"Full Model    - R²: {full_metrics['r2']:.4f}, MSE: {full_metrics['mse']:.4f}")
            print(f"Filtered Model- R²: {filtered_metrics['r2']:.4f}, MSE: {filtered_metrics['mse']:.4f}")
            print(f"Improvement in R²: {filtered_metrics['r2'] - full_metrics['r2']:.4f}")
            print(f"Reduction in MSE: {full_metrics['mse'] - filtered_metrics['mse']:.4f}")
            


if __name__ == "__main__":
    main()
