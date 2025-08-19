# Accelerate Flash Removal of PFAS from Soil by Human-guided Bayesian Optimization and Interpretable Machine Learning


A comprehensive machine learning pipeline for PFAS (Per- and polyfluoroalkyl substances) soil treatment optimization using Gaussian Process Regression, XGBoost constraint models, and Human-Guided Bayesian Optimization for intelligent experimental design.

## Overview

This project provides an integrated solution for PFAS soil treatment research with:
- **Gaussian Process Regression** for PFAS content prediction with uncertainty quantification
- **XGBoost Constraint Model** for initial resistance prediction
- **MBNN (Multi-Branch Neural Network)** for PFAS treatment efficiency prediction using experimental, chemical, and molecular graph features
- **HGBO (Human-Guided Bayesian Optimization)** for intelligent experimental design and next-point recommendation
- **Model Interpretability** using SHAP analysis and Partial Dependence Plots
- **Visualizations** with real-time heatmaps and sampling visualization

## Project Structure

```
HGBO/
├── PFAS soil Database.xlsx    # Experimental dataset
├── Processing.py              # Data preprocessing utilities
├── XG_R_constraint_model.py   # XGBoost constraint model for resistance prediction  
├── GP.py                      # Gaussian Process model for PFAS prediction
├── Main Loop.ipynb            # HGBO main loop
└── MBNN.ipynb                 # Multi-Branch Neural Network
```

## Installation

Clone or download the repository and install dependencies:

```bash
cd HGBO
pip install -r requirements.txt
```

## Usage Workflow

The complete HGBO workflow follows this sequence:

### Step 1: Data Preprocessing (Processing.py)

Run the data preprocessing module to clean and prepare the experimental dataset. This module handles:
- Data loading from Excel files
- Missing value imputation
- Outlier detection using Local Outlier Factor
- Feature scaling and encoding
- Data validation and cleaning

The preprocessing creates the foundational dataset used by all subsequent models.

### Step 2: Train XGBoost Constraint Model (XG_R_constraint_model.py)

Train the XGBoost regression model for resistance prediction. This model serves as a constraint function in the optimization process:
- Predicts initial resistance values based on material properties
- Performs hyperparameter tuning using grid search
- Provides constraint validation for the HGBO loop
- Generates feature importance analysis
- Saves the trained model for deployment

### Step 3: Train Gaussian Process Model (GP.py)

Train the Gaussian Process regression model for PFAS content prediction:
- Implements enhanced GP kernels (Matérn + RationalQuadratic + WhiteKernel)
- Performs automatic hyperparameter optimization
- Provides uncertainty quantification for predictions
- Conducts SHAP interpretability analysis
- Generates partial dependence plots
- Saves the trained model for optimization

### Step 4: Run HGBO Main Loop (Main Loop.ipynb)

Execute the Human-Guided Bayesian Optimization process:
- Loads pre-trained GP and XGBoost models
- Implements interactive experimental design loop
- Generates constraint-aware experimental recommendations
- Collects human feedback and scores
- Updates models based on new experimental data
- Provides real-time visualizations including sampling plots and GP heatmaps
- Adapts acquisition function based on user preferences

## Core Components

### Data Processing Module

The Processing.py module provides comprehensive data preprocessing capabilities:
- Handles missing values in both numerical and categorical features
- Applies Local Outlier Factor for outlier detection
- Performs one-hot encoding for categorical variables
- Standardizes numerical features
- Validates data quality and consistency

### XGBoost Constraint Model

The XG_R_constraint_model.py implements a constraint model for resistance prediction:
- Uses XGBoost regression for accurate resistance forecasting
- Incorporates grid search for optimal hyperparameter selection
- Provides feature importance analysis
- Validates predictions through residual analysis
- Ensures constraint satisfaction in the optimization loop

### Gaussian Process Model

The GP.py module implements advanced Gaussian Process regression:
- Enhanced kernel design combining multiple kernel types
- Automatic hyperparameter optimization with multiple restarts
- Uncertainty quantification for informed decision making
- SHAP-based model interpretability analysis
- 1D and 2D partial dependence plot generation
- Visualization capabilities

### Multi-Branch Neural Network (MBNN)

The MBNN implements a three-stream neural network architecture for PFAS treatment efficiency prediction:
- **Experimental Branch**: Processes experimental conditions and operational parameters
- **Chemical Branch**: Handles molecular descriptors and chemical properties of PFAS compounds
- **Graph Branch**: Incorporates molecular structure information through graph neural networks
- **Fusion Layer**: Intelligently combines outputs from all three branches for final prediction
- **SHAP Interpretability**: Provides feature importance analysis across all input modalities
- Branch importance analysis to understand contribution of different data types

### HGBO Main Loop

The Main Loop.ipynb implements the Human-Guided Bayesian Optimization system:
- Interactive experimental design with human feedback integration
- Acquisition function optimization using differential evolution
- Real-time model updates based on new experimental results
- Constraint-aware sampling through XGBoost predictions
- Dynamic visualization including PCA-based sampling plots and GP uncertainty heatmaps
- Adaptive optimization strategy based on user scoring

## Key Features

### Advanced Machine Learning Pipeline
- Multi-model approach combining GP prediction, XGBoost constraints, and MBNN efficiency prediction
- Multi-modal learning through MBNN combining experimental, chemical, and molecular graph data
- Uncertainty quantification for informed experimental decisions
- Real-time model updates incorporating new experimental data
- Constraint-aware optimization ensuring feasible experimental conditions

### Human-Guided Optimization
- Interactive feedback collection through user scoring system
- Adaptive acquisition function responding to human preferences
- Real-time visualization showing optimization progress
- Intelligent stopping criteria based on user feedback

### Model Interpretability
- SHAP analysis revealing feature contributions to predictions
- Partial dependence plots showing individual feature effects
- Feature importance ranking across different models
- Interactive visualizations supporting model exploration

## Data Format

### Input Features
- Material Name: Categorical variable specifying soil/additive material type
- Weight_ratio: Categorical variable indicating material mixing ratios
- Initial total mass: Numerical variable representing total mass in milligrams
- Init. Res.(Ω): Numerical variable for initial electrical resistance in Ohms
- Init. Pulse Voltage: Numerical variable for initial pulse voltage in Volts

### Target Variables
- PFAS Content: Primary target representing PFAS concentration (for GP model)
- F ion Content: Secondary target for fluoride ion concentration (for GP model)  
- Treatment Efficiency: PFAS removal efficiency percentage (for MBNN model)

## Customization Options

### Gaussian Process Kernels
The GP model supports various kernel modifications including length scale adjustments, boundary constraints, and smoothness parameters.

### Acquisition Function
The HGBO system allows customization of the acquisition function including different combinations of exploitation and exploration terms.

## Contact

For questions, issues, or research collaboration opportunities, please contact Prof. Lin: linjian@missouri.edu

---

**Advancing PFAS remediation research through intelligent experimental design**
