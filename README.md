# NYC Taxi Trip Duration Prediction

An comprehensive machine learning project that predicts New York City taxi trip durations using advanced regression techniques, including deep neural networks and ensemble methods. This project demonstrates end-to-end ML pipeline development from data exploration to model comparison and evaluation.

## Project Overview

This project analyzes the NYC Taxi Trip Duration dataset to build predictive models that can accurately estimate taxi trip durations based on multiple features including pickup/dropoff locations, temporal patterns, passenger information, and engineered distance metrics. The project implements both traditional machine learning algorithms and deep learning approaches with systematic hyperparameter optimization.

## Dataset

The project uses the NYC Taxi Trip Duration dataset which includes:
- **Pickup/Dropoff coordinates** (latitude, longitude)
- **Pickup/Dropoff timestamps** 
- **Passenger count** (1-6 passengers)
- **Vendor ID** (taxi company identifier)
- **Store and forward flag** (trip recording method)
- **Trip duration** (target variable in seconds)

### Dataset Statistics
- **Training samples**: ~1.4 million records
- **Features**: 11 original features + engineered features
- **Target**: Trip duration (continuous regression problem)

## Key Features & Methodology

### 📊 Comprehensive Exploratory Data Analysis (EDA)
- **Statistical Analysis**: Distribution analysis, correlation matrices, outlier detection
- **Temporal Patterns**: Trip frequency analysis over time, seasonal trends
- **Geospatial Analysis**: Pickup/dropoff location clustering and geographic visualization
- **Univariate Analysis**: Custom visualization functions for data distribution and Q-Q plots
- **Data Quality Assessment**: Missing value analysis and data type validation

### 🔧 Advanced Feature Engineering
- **Distance Calculations**:
  - **Haversine Distance**: Great-circle distance between pickup/dropoff points
  - **Manhattan Distance Approximation**: City-block distance estimation
  - **Bearing/Direction**: Angular direction of travel
- **Temporal Feature Extraction**:
  - Year, month, day, hour, minute, second components
  - Separate pickup and dropoff time features
- **Data Cleaning**:
  - Outlier removal using statistical methods (mean ± 2σ)
  - Passenger count validation (1-6 passengers only)
  - Extreme duration filtering

### 🧹 Data Preprocessing Pipeline
- **Feature Scaling**: MinMaxScaler for numerical features
- **Categorical Encoding**: OneHotEncoder for categorical variables
- **Train-Test Split**: 80/20 split with stratification
- **Data Type Conversion**: TensorFlow tensor conversion for neural networks

### 🤖 Model Implementation

#### Deep Neural Networks (TensorFlow/Keras)
1. **Simple Baseline Model**
   - Architecture: 100 → 10 → 1 neurons
   - Activation: ReLU, Linear output
   - Optimizer: Adam

2. **Advanced Deep Model (small_model)**
   - Architecture: 256 → 128 → 64 → 1 neurons
   - Regularization: Batch Normalization, Dropout (0.1)
   - Early Stopping with patience=3

3. **Optimized Deep Model (small_model2)**
   - Architecture: 256 → 128 → 64 → 1 neurons
   - Reduced Dropout: 0.05
   - Extended training: 15 epochs

4. **L2 Regularized Models (small_model3 & small_model4)**
   - Architecture: 128 → 64 → 32 → 1 neurons
   - L2 Regularization: 1e-4 and 1e-3 respectively
   - Batch Normalization + Dropout combination

#### Traditional Machine Learning Models
- **Decision Tree Regressor**: With depth and sample tuning
- **Random Forest**: Ensemble with bootstrap aggregating
- **Gradient Boosting**: Sequential weak learner combination
- **XGBoost**: Optimized gradient boosting implementation

### 🔍 Hyperparameter Optimization
- **RandomizedSearchCV**: Automated hyperparameter tuning
- **Cross-Validation**: 3-fold CV for robust evaluation
- **Parameter Grids**: Comprehensive search spaces for each algorithm
- **Performance Metrics**: Mean Absolute Error (MAE) optimization

### 📈 Model Evaluation & Visualization
- **Custom Loss Curve Plotting**: Training/validation loss visualization
- **Model Architecture Visualization**: TensorFlow model plotting
- **Performance Comparison**: Systematic MAE comparison across models
- **Random Prediction Demos**: Sample prediction analysis

## Technical Implementation

### Libraries & Dependencies
```python
# Data Processing
pandas, numpy

# Visualization  
matplotlib, seaborn

# Machine Learning
scikit-learn, xgboost

# Deep Learning
tensorflow, keras

# Statistical Analysis
statsmodels

# Utilities
warnings, os, IPython
```

### Custom Functions
- `set_seed()`: Reproducible random state management
- `haversine_array()`: Vectorized great-circle distance calculation
- `dummy_manhattan_distance()`: Manhattan distance approximation
- `bearing_array()`: Travel direction calculation
- `preprocessing_data()`: Complete data preprocessing pipeline
- `univariate_analysis()`: Statistical distribution visualization
- `plot_loss_curves()`: Training progress visualization

## Results & Performance

The project systematically compares multiple model architectures:

### Neural Network Performance
- **Simple Model**: Basic 3-layer architecture baseline
- **Deep Models**: Progressive complexity with regularization
- **Optimization**: Early stopping, batch normalization, dropout

### Traditional ML Performance
- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost
- **Tree-based Models**: Decision trees with various configurations
- **Hyperparameter Tuning**: Automated optimization via RandomizedSearchCV

### Evaluation Metrics
- **Primary Metric**: Mean Absolute Error (MAE)
- **Validation Strategy**: Hold-out validation + Cross-validation
- **Model Selection**: Performance-based best model identification

## Project Structure

```
├── NYC_2016_Taxi_Prediction.ipynb    # Main analysis notebook (79 cells)
├── NYC_2016_Taxi_Prediction_Report.pdf        # Detailed project report
├── README.md                   # This documentation
├── .gitignore                  # Version control exclusions
└── .venv/                      # Python virtual environment
```

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost statsmodels ipython
```

### Running the Project
1. **Clone the repository**
2. **Extract the dataset**: `https://www.kaggle.com/c/nyc-taxi-trip-duration)`
3. **Open the notebook**: Launch `NYC_2016_Taxi_Prediction.ipynb` in Jupyter
4. **Execute cells sequentially**: Run all cells for complete analysis
5. **Review results**: Compare model performances and visualizations

### Dataset Setup
The notebook expects the dataset to be extracted to:
```
nyc-taxi-trip-duration/
├── train.csv
└── test.csv
```

## Key Insights

### Data Characteristics
- **Temporal Patterns**: Clear daily and seasonal trip patterns
- **Geographic Distribution**: Concentrated in Manhattan with outer borough coverage
- **Duration Distribution**: Right-skewed with log-normal characteristics
- **Feature Correlations**: Strong relationship between distance metrics and trip duration

### Model Performance
- **Deep Learning**: Effective with proper regularization and architecture design
- **Ensemble Methods**: Robust performance across different data patterns
- **Feature Engineering**: Significant impact on model accuracy
- **Hyperparameter Tuning**: Measurable performance improvements

## Academic Context

**Course**: Introduction to Machine Learning (IML) 2025  
**Author**: Ansh Madan  
**Institution**: Final Project Submission  
**Focus Areas**: Regression, Feature Engineering, Deep Learning, Model Comparison

## Technical Highlights

- **Reproducible Research**: Comprehensive seed management for consistent results
- **Production-Ready Code**: Modular functions and clean implementation
- **Visualization Excellence**: Professional plots with custom styling
- **Performance Optimization**: Efficient data processing and model training
- **Documentation**: Extensive comments and markdown explanations

## Future Enhancements

- **Real-time Prediction API**: Deploy best model as web service
- **Additional Features**: Weather data, traffic patterns, event information
- **Advanced Architectures**: LSTM for temporal patterns, CNN for spatial features
- **Ensemble Stacking**: Combine multiple model predictions
- **Production Monitoring**: Model drift detection and retraining pipelines

---

*This project demonstrates comprehensive machine learning methodology from data exploration through model deployment, showcasing both traditional and modern approaches to regression problems.*
