# Dream11 Fantasy Team Predictor

> ⚠️ **USER INPUT ALERT**  
> - When prompted for team names, **enter the team code exactly as shown** (e.g., `RCB`, `MI`, `CSK`, etc.).  
> - When prompted for venue, **enter the official home venue for the home team** (e.g., for RCB enter `Bengaluru`, for MI enter `Mumbai`, for CSK enter `Chennai`, etc.).  
> - Example:  
>   - For RCB as home team, enter `RCB` for Home Team and `Bengaluru` for Venue.  
>   - For MI as home team, enter `MI` for Home Team and `Mumbai` for Venue.  
> - **Incorrect or misspelled entries may result in errors or inaccurate predictions.**

⚠️ **IMPORTANT ALERT**  
Please ensure to:
1. Download the latest `SquadPlayerNames_IndianT20League_Dup.xlsx` dataset daily at 7 PM after team lineups are announced
2. Replace the existing file in your `data/` directory with the newly downloaded version
3. This is crucial for accurate team predictions as it contains the most recent playing XI information

> All datasets used in this project except the SquadPlayerNames dataset were prepared by myself to ensure quality and accuracy.

## Overview
This project is developed as part of the Foundations of Machine Learning (FoML) course. The system leverages advanced machine learning techniques and ensemble methods to predict optimal fantasy cricket teams for the Indian T20 League based on player statistics, match conditions, and historical performance data. The system now features enhanced algorithms, improved feature engineering, and a more balanced approach between ML predictions and expert knowledge.

## Technical Implementation

### Machine Learning Models
- **Enhanced Ensemble Approach**
  - Multiple algorithms with weighted averaging based on cross-validation performance
  - Automatic model selection and hyperparameter optimization

- **Core Algorithms**
  - XGBoost Regressor - Gradient boosted decision trees for accurate predictions
  - Random Forest Regressor - Robust ensemble of decision trees
  - Gradient Boosting Regressor - Sequential tree building for error reduction
  - AdaBoost Regressor - Adaptive boosting for focusing on difficult predictions
  - ElasticNet - Linear regression with L1 and L2 regularization
  - Support Vector Regression (SVR) - Non-linear regression with kernel methods
  - Multi-layer Perceptron (MLP) - Neural network for complex pattern recognition
  - TensorFlow Neural Network (optional) - Deep learning for advanced pattern recognition

- **Advanced Feature Engineering**
  - Role-specific performance metrics
  - Form and consistency tracking
  - Venue and opposition analysis
  - Advanced batting and bowling efficiency metrics
  - Boundary percentage and non-boundary strike rate calculations

### Fantasy Points Calculation System
1. **Batting Points**
   - Base Points: 1 point per run
   - Boundary Bonus: 
     - Four: 1 point
     - Six: 2 points
   - Strike Rate Bonuses (minimum 10 balls):
     - SR ≥ 170: 6 points
     - SR ≥ 150: 4 points
     - SR ≥ 130: 2 points
   - Milestone Bonuses:
     - Century (100 runs): 16 points
     - 75 runs: 8 points
     - Half-century (50 runs): 4 points
     - 25 runs: 2 points

2. **Bowling Points**
   - Wickets: 30 points each
   - Special Wicket Bonuses:
     - 5 wicket haul: 8 bonus points
     - 4 wickets: 4 bonus points
     - 3 wickets: 4 bonus points
   - Maiden Over: 4 points
   - Economy Rate Bonuses (minimum 2 overs):
     - ≤ 6.0: 6 points
     - ≤ 7.0: 4 points
     - ≤ 8.0: 2 points

3. **Fielding Points**
   - Catch: 8 points
   - Stumping: 12 points
   - Run-out (Direct): 12 points
   - Run-out (Assist): 6 points

## Project Structure
```
dream11-predictor/
├── app/
│   ├── catboost_info/
│   │   ├── catboost_training.json
│   │   ├── learn/
│   │   ├── learn_error.tsv
│   │   ├── time_left.tsv
│   │   └── tmp/
│   └── main.py                # Main application entry point
├── catboost_info/
│   ├── catboost_training.json
│   ├── learn/
│   │   └── events.out.tfevents
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   └── tmp/
├── data/
│   ├── MATCH_DATA_COMBINED_DATASET.xlsx
│   ├── MATCH_METADATA.xlsx
│   ├── SquadPlayerNames_IndianT20League_Dup.xlsx
│   ├── captaincy_priority.py
│   └── credits_reference_with_priority.xlsx
├── outputs/
│   ├── final_team_output.csv
│   └── logs/
│       └── selection_log.txt
├── src/
│   ├── __init__.py
│   ├── data_standardizer.py
│   ├── enhanced_model_predictor.py
│   ├── fantasy_point_calculator.py
│   ├── model_predictor.py
│   ├── neural_network_model.py
│   ├── recent_form_generator.py
│   ├── strategy_engine.py
│   ├── team_selector.py
│   └── utils.py
└── requirements.txt
```

## Features
- **Enhanced Data Processing Pipeline**
  - Automated data cleaning and standardization
  - Advanced feature engineering for player performance metrics
  - Historical performance analysis with improved pattern recognition
  - Intelligent handling of missing data and outliers
  - Support for X-Factor substitutes in team selection

- **Improved Team Selection Strategy**
  - Credit-based optimization (100 points limit)
  - Balanced role distribution:
    - 1-2 Wicket-keepers (WK)
    - 3-5 Batsmen (BAT)
    - 3-5 Bowlers (BOWL)
    - 1-2 All-rounders (ALL)
  - Intelligent Captain (2x points) and Vice-Captain (1.5x points) selection
  - Opposition analysis integration with venue-specific performance metrics
  - Enhanced backup player selection logic

- **Advanced Performance Analytics**
  - Role-specific performance metrics
  - Form and consistency tracking
  - Venue and opposition-specific analysis
  - Advanced batting metrics (boundary percentage, non-boundary strike rate)
  - Advanced bowling metrics (bowling average, bowling strike rate, dot ball estimation)

## Datasets
1. **MATCH_DATA_COMBINED_DATASET.xlsx**
   - Match statistics and performance metrics
   - Historical player performance data

2. **SquadPlayerNames_IndianT20League_Dup.xlsx**
   - Daily squad information
   - Player availability data

3. **MATCH_METADATA.xlsx**
   - Venue information
   - Match conditions
   - Team-specific data

4. **credits_reference_with_priority.xlsx**
   - Player credit values
   - Strategic priority rankings

## Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dream11-predictor.git
cd dream11-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the prediction system:
```bash
python app/main.py
```

4. Enter the required information:
   - Match ID
   - Home Team
   - Away Team
   - Venue
   - Toss Winner
   - Toss Decision

The predicted team will be saved in `outputs/final_team_output.csv`

## Dependencies
- pandas - Data manipulation and analysis
- numpy - Numerical computing
- scikit-learn - Machine learning algorithms and utilities
- matplotlib - Data visualization
- seaborn - Statistical data visualization
- jupyter - Interactive development
- xgboost - Gradient boosting framework (optional)
- tensorflow - Deep learning framework (optional)
- catboost - Gradient boosting on decision trees (optional)
- lightgbm - Gradient boosting framework (optional)

## System Improvements

### Enhanced Logging System
- **Intelligent Log Filtering** - Custom ModuleFilter class to prioritize important messages
- **Dual Output Channels** - Detailed logs to file for debugging, filtered logs to console for user experience
- **Team Selection Summary** - Clearly formatted output of final team composition
- **Error Handling** - Comprehensive error logging with appropriate verbosity levels
- **Library Warning Suppression** - Filtering of non-essential warnings from TensorFlow and other libraries

### Performance Optimizations
- **Efficient Data Processing** - Improved data handling for faster team selection
- **Reduced Redundancy** - Elimination of duplicate code in model training and prediction
- **Memory Management** - Better handling of large datasets with proper indexing
- **Exception Handling** - Robust error recovery with informative user feedback

## Acknowledgments
This project was developed as part of the Foundations of Machine Learning (FoML) course project. Special thanks to the course instructors for their guidance and support throughout the development process.