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

Failing to update this file may result in predictions based on outdated squad information.

## Overview
This project is developed as part of the Foundations of Machine Learning (FoML) course. The system leverages machine learning to predict optimal fantasy cricket teams for the Indian T20 League based on player statistics, match conditions, and historical performance data.

## Technical Implementation

### Machine Learning Models
- **XGBoost Regressor**
  - Player performance prediction
  - Feature importance analysis
  - Historical data pattern recognition

- **Ensemble Methods**
  - Weighted averaging of predictions
  - Cross-validation for model selection
  - Hyperparameter optimization

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
│   └── main.py                # Main application entry point
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── data_standardizer.py
│   ├── fantasy_point_calculator.py
│   ├── model_predictor.py
│   ├── recent_form_generator.py
│   ├── strategy_engine.py
│   ├── team_selector.py
│   └── utils.py
├── data/
│   ├── credits_reference_with_priority.xlsx
│   ├── MATCH_DATA_COMBINED_DATASET.xlsx
│   ├── MATCH_METADATA.xlsx
│   └── SquadPlayerNames_IndianT20League_Dup.xlsx
├── outputs/
│   ├── logs/
│   └── final_team_output.csv
└── requirements.txt
```

## Features
- **Data Processing Pipeline**
  - Automated data cleaning and standardization
  - Feature engineering for player performance metrics
  - Historical performance analysis

- **Team Selection Strategy**
  - Credit-based optimization (100 points limit)
  - Role distribution:
    - 1-2 Wicket-keepers (WK)
    - 3-5 Batsmen (BAT)
    - 3-5 Bowlers (BOWL)
    - 1-2 All-rounders (ALL)
  - Captain (2x points) and Vice-Captain (1.5x points)
  - Opposition analysis integration

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
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Acknowledgments
This project was developed as part of the Foundations of Machine Learning (FoML) course project. Special thanks to the course instructors for their guidance and support throughout the development process.

