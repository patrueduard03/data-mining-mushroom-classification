# 🍄 Mushroom Classification Project

An academic data mining project that uses machine learning to classify mushrooms as edible or poisonous based on their physical characteristics, featuring a comprehensive Jupyter notebook analysis and an interactive web application.

## Table of Contents

- [Academic Information](#academic-information)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
  - [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Analysis](#1-data-analysis-mushroom_classificationipynb)
  - [2. Model Development](#2-model-development)
  - [3. Web Application](#3-web-application)
- [Quick Start](#quick-start)
  - [1. Setup Environment](#1-setup-environment)
  - [2. Run Jupyter Analysis](#2-run-jupyter-analysis)
  - [3. Launch Web Application](#3-launch-web-application)
- [Model Performance](#model-performance)
  - [Key Insights](#key-insights)
- [Web Interface Features](#web-interface-features)
- [Safety Warning](#safety-warning)
- [Technical Stack](#technical-stack)
- [Future Enhancements](#future-enhancements)

## Academic Information

**Institution:** West University of Timisoara  
**Faculty:** Mathematics and Informatics  
**Course:** Data Mining Project Big Data - Year 1  
**Project Authors:** Patru Gheorghe Eduard and Mihoc Cristian
**Year:** 2025

## Project Overview

This project demonstrates the complete machine learning pipeline from data exploration to model deployment:

1. **Data Analysis** - Comprehensive exploratory data analysis in Jupyter notebook
2. **Model Development** - Advanced CatBoost classifier with hyperparameter optimization
3. **Web Interface** - Interactive Flask application for real-time mushroom classification
4. **Model Interpretability** - SHAP analysis and feature importance visualization

## Dataset

The project uses a comprehensive mushroom dataset containing:
- **8,124 mushroom samples** with 20 physical characteristics
- **Features include:** cap properties, gill characteristics, stem attributes, habitat, and seasonal information
- **Target variable:** Binary classification (edible vs poisonous)
- **Data source:** Publicly available mushroom classification dataset

### Key Features
- Cap diameter, shape, surface, and color
- Gill attachment, spacing, and color  
- Stem height, width, root type, surface, and color
- Environmental factors: habitat and season
- Physical properties: bruising, ring presence, spore print color

## Project Structure

```
├── mushroom_classification.ipynb    # Main analysis notebook
├── src/
│   ├── app.py                      # Flask web application
│   └── templates/index.html        # Modern web interface
├── models/
│   ├── catboost_model.cbm         # Trained CatBoost model
│   └── best_params.json           # Optimized hyperparameters
├── data/
│   ├── mushroom.csv               # Original dataset
│   └── mushroom_classification_results.csv  # Detailed predictions
├── figures/                        # Analysis visualizations
├── utils/
│   ├── feature_mappings.json      # Feature encoding mappings
│   └── examine_model.py           # Model utilities
└── requirements.txt               # Python dependencies
```

## Methodology

### 1. Data Analysis (`mushroom_classification.ipynb`)
- **Exploratory Data Analysis:** Missing values, distributions, correlations
- **Data Preprocessing:** Feature encoding, train/test split, categorical handling
- **Baseline Model:** Logistic Regression with one-hot encoding
- **Advanced Model:** CatBoost with Optuna hyperparameter optimization
- **Model Evaluation:** Confusion matrix, ROC curves, classification metrics
- **Interpretability:** SHAP values, permutation importance, feature analysis

### 2. Model Development
- **Algorithm:** CatBoost Classifier (gradient boosting)
- **Optimization:** Optuna for hyperparameter tuning (10 trials)
- **Features:** Native categorical feature handling
- **Performance:** High accuracy with comprehensive evaluation metrics
- **Persistence:** Model saved as `catboost_model.cbm` for deployment

### 3. Web Application
- **Backend:** Flask framework with prediction API
- **Frontend:** Modern, responsive HTML/CSS/JavaScript interface
- **Features:**
  - Interactive form with dropdown menus and input fields
  - Real-time classification with confidence scores
  - Random value generator for testing
  - Educational warnings about mushroom safety
  - Modern dark theme with glassmorphism effects

## Quick Start

### 1. Setup Environment
```bash
# Clone or download the project
cd data-mining-mushroom-classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Jupyter Analysis
```bash
# Open the main analysis notebook
jupyter notebook mushroom_classification.ipynb
```

### 3. Launch Web Application
```bash
# Start the Flask application
python src/app.py

# Open browser to http://localhost:5000 or the port available on the local machine
```

## Model Performance

The CatBoost classifier achieves excellent performance on the mushroom classification task:
- **High accuracy** on test data
- **Robust feature handling** with categorical variables
- **Interpretable predictions** through SHAP analysis
- **Confidence scoring** for prediction reliability

### Key Insights
- Most important features for classification include spore print color, gill characteristics, and cap properties
- Model provides confidence levels (High/Medium/Low) for prediction reliability
- SHAP analysis reveals feature contributions for individual predictions

## Web Interface Features

The interactive web application provides:
- **User-friendly form** with all 20 mushroom characteristics
- **Instant classification** with probability scores
- **Confidence indicators** (Very High, High, Medium, Low)
- **Random value generator** for quick testing
- **Educational warnings** emphasizing safety
- **Modern design** with responsive layout

## Safety Warning

⚠️ **EDUCATIONAL PURPOSE ONLY**  
This project is for academic demonstration. **NEVER** use any automated tool to identify wild mushrooms for consumption. Mushroom identification requires expert knowledge, and misidentification can be fatal.

## Technical Stack

- **Data Analysis:** Python, Pandas, NumPy, Matplotlib
- **Machine Learning:** CatBoost, Scikit-learn, Optuna
- **Model Interpretation:** SHAP, Permutation Importance
- **Web Framework:** Flask
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Deployment:** Local development server
