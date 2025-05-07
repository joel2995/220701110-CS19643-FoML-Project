# Dream11 Fantasy Team Predictor

> ⚠️ **USER INPUT ALERT**  
> - When prompted for team names, **enter the team code exactly as shown** (e.g., RCB, MI, CSK, etc.).  
> - When prompted for venue, **enter the official home venue for the home team** (e.g., for RCB enter Bengaluru, for MI enter Mumbai, for CSK enter Chennai, etc.).  
> - Example:  
>   - For RCB as home team, enter RCB for Home Team and Bengaluru for Venue.  
>   - For MI as home team, enter MI for Home Team and Mumbai for Venue.  
> - **Incorrect or misspelled entries may result in errors or inaccurate predictions.**

⚠️ **IMPORTANT ALERT**  
Please ensure to:
1. Download the latest `SquadPlayerNames.xlsx` dataset daily at 7 PM after team lineups are announced  
2. Replace the existing file in your `data/` directory with the newly downloaded version  
3. This is crucial for accurate team predictions as it contains the most recent playing XI information

> All other datasets used in this project were custom-prepared by the author to ensure quality and accuracy.

---

## Overview

This project is developed as part of the Foundations of Machine Learning (FoML) course. The system leverages advanced machine learning techniques and ensemble methods to predict optimal fantasy cricket teams for the Indian T20 League based on player statistics, match conditions, and historical performance data. The system features enhanced algorithms, improved feature engineering, and a balanced approach between ML predictions and domain expertise.

---

## Technical Implementation

### Machine Learning Models

- **Enhanced Ensemble Approach**
  - Multiple algorithms with weighted averaging based on cross-validation performance
  - Automatic model selection and hyperparameter optimization

- **Core Algorithms**
  - XGBoost Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - AdaBoost Regressor
  - ElasticNet
  - Support Vector Regression (SVR)
  - Multi-layer Perceptron (MLP)
  - TensorFlow Neural Network (optional)

- **Advanced Feature Engineering**
  - Role-specific performance metrics
  - Form and consistency tracking
  - Venue and opposition analysis
  - Batting and bowling efficiency metrics
  - Boundary percentage and non-boundary strike rate calculations

---

### Fantasy Points Calculation System

1. **Batting Points**
   - 1 point per run
   - Boundary Bonus: 4s = 1 pt, 6s = 2 pts
   - SR Bonuses (min 10 balls):
     - SR ≥ 170: 6 pts
     - SR ≥ 150: 4 pts
     - SR ≥ 130: 2 pts
   - Milestones:
     - 100: 16 pts, 75: 8 pts, 50: 4 pts, 25: 2 pts

2. **Bowling Points**
   - 30 pts per wicket
   - Special Wickets:
     - 3/4/5 wickets: 4/4/8 pts
   - Maiden: 4 pts
   - Economy (min 2 overs):
     - ≤ 6.0: 6 pts
     - ≤ 7.0: 4 pts
     - ≤ 8.0: 2 pts

3. **Fielding Points**
   - Catch: 8 pts
   - Stumping: 12 pts
   - Run-out (Direct): 12 pts, (Assist): 6 pts

---

## Project Structure

```
dream11-predictor/
├── app/
│   ├── catboost_info/
│   └── main.py                     # Entry point for running the model and predictions

├── catboost_info/                 # Additional CatBoost model information

├── data/                          # All input Excel datasets
│   ├── captaincy_priority.xlsx
│   ├── credits_reference.xlsx
│   ├── MATCH_DATA_INPUT.xlsx
│   ├── MATCH_METADATA_INPUT.xlsx
│   └── SquadPlayerNames.xlsx

├── outputs/                       # Logs and final output teams
│   ├── logs/
│   │   └── selection_log.txt
│   └── final_team_output.xlsx

├── src/                           # Core source code and logic
│   ├── __init__.py
│   ├── data_standardizer.py
│   ├── enhanced_model.py
│   ├── fantasy_point_calculator.py
│   ├── model_predictor.py
│   ├── neural_network_model.py
│   ├── recent_form_generator.py
│   ├── strategy_engine.py
│   ├── team_selector.py
│   └── utils.py

├── requirements.txt               # Required dependencies

└── README.md                      # Project overview and instructions
```

---

## Features

### Data Processing
- Automated cleaning and standardization
- Advanced feature extraction
- Historical performance and pattern analysis
- Handles missing data and outliers
- X-Factor substitute player support

### Team Selection Strategy
- 100 credit limit optimization
- Role-wise player balance:
  - 1–2 WK, 3–5 BAT, 3–5 BOWL, 1–2 ALL
- Smart Captain/Vice-Captain selection
- Venue & opponent-based analysis
- Backup player logic for flexibility

### Analytics
- Batting metrics: Boundary%, NBSR
- Bowling metrics: Economy, Strike Rate, Dot Balls
- Consistency & Form trackers

---

## Datasets

1. **MATCH_DATA_INPUT.xlsx**  
   - Custom-prepared match statistics and player performance data

2. **SquadPlayerNames.xlsx**  
   - Daily squad list (downloaded externally)

3. **MATCH_METADATA_INPUT.xlsx**  
   - Venue and match details (custom-prepared)

4. **credits_reference.xlsx**  
   - Credit values and rankings (custom-prepared)

5. **captaincy_priority.xlsx**  
   - Priority logic for captain/vice-captain selection (custom-prepared)

---

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

4. Enter the required match details:
   - Match ID
   - Home Team
   - Away Team
   - Venue
   - Toss Winner
   - Toss Decision

The final fantasy team will be saved at:
`outputs/final_team_output.xlsx`

---

## Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and utilities
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **jupyter** - Interactive development
- **xgboost** - Gradient boosting framework (optional)
- **tensorflow** - Deep learning framework (optional)
- **catboost** - Gradient boosting on decision trees (optional)
- **lightgbm** - Gradient boosting framework (optional)

---

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

---

## Acknowledgments

This project was developed for the Foundations of Machine Learning (FoML) course. Thanks to the instructors for their guidance and support.
```