import pandas as pd
import sys
import os
import logging

# Set base directory for Docker environment
base_dir = 'e:\\220701110-FoML-Project'

# Set up logging to file and console
log_dir = os.path.join(base_dir, 'outputs', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'selection_log.txt')

# Configure logging - File logging remains detailed for debugging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

# Console handler only shows INFO level messages from main module
# This filters out warnings from TensorFlow and other libraries
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

# Create a filter to only show messages from the main module and team_selector
class ModuleFilter(logging.Filter):
    def filter(self, record):
        # Always show errors
        if record.levelno >= logging.ERROR:
            return True
            
        # Always show the team selection summary and related information
        if "[TEAM SELECTION SUMMARY]" in record.getMessage() or \
           "Team composition:" in record.getMessage() or \
           "Players from each team:" in record.getMessage() or \
           "Total credits used:" in record.getMessage() or \
           "Final 11:" in record.getMessage() or \
           "Backups:" in record.getMessage() or \
           "Captain:" in record.getMessage() or \
           "Vice-Captain:" in record.getMessage() or \
           "Final Team (Submission Format)" in record.getMessage():
            return True
            
        # Allow specific initialization messages
        allowed_messages = [
            "Initializing TeamSelector",
            "Loading data files",
            "Data files loaded and standardized successfully",
            "Initializing components",
            "Components initialized successfully",
            "Predicting the best team",
            "Starting prediction process",
            "Loading squad data for match ID"
        ]
        for msg in allowed_messages:
            if msg in record.getMessage():
                return True
        return False

# Apply the filter to console output only
console_handler.addFilter(ModuleFilter())

logger = logging.getLogger()
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Suppress TensorFlow and other library warnings
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

# Add src to sys path for import
sys.path.append(os.path.join(base_dir, 'src'))
from team_selector import TeamSelector

def get_team_input(prompt):
    VALID_TEAMS = ['DC', 'CSK', 'MI', 'LSG', 'PBKS', 'RCB', 'GT', 'SRH', 'KKR', 'RR']
    team_prompt = f"{prompt}({'/'.join(VALID_TEAMS)}): "
    while True:
        team = input(team_prompt).upper()
        if team in VALID_TEAMS:
            return team
        print(f"[ERROR] Invalid team. Please choose from: {', '.join(VALID_TEAMS)}")

def get_venue_input():
    VALID_VENUES = [
        'Ahmedabad', 'Bengaluru', 'Chandigarh', 'Chennai', 'Delhi', 
        'Guwahati', 'Hyderabad', 'Jaipur', 'Kolkata', 'Lucknow', 
        'Mumbai', 'Visakhapatnam'
    ]
    venue_prompt = f"Enter Venue({'/'.join(VALID_VENUES)}): "
    while True:
        venue = input(venue_prompt).title()
        if venue in VALID_VENUES:
            return venue
        print(f"[ERROR] Invalid venue. Please choose from: {', '.join(VALID_VENUES)}")

def get_toss_decision():
    VALID_DECISIONS = ['Bat', 'Bowl']
    decision_prompt = f"Enter Toss Decision({'/'.join(VALID_DECISIONS)}): "
    while True:
        decision = input(decision_prompt).title()
        if decision in VALID_DECISIONS:
            return decision
        print(f"[ERROR] Invalid decision. Please choose from: {', '.join(VALID_DECISIONS)}")

def get_toss_winner_input(home_team, away_team):
    playing_teams = [home_team, away_team]
    toss_prompt = f"Enter Toss Winner({'/'.join(playing_teams)}): "
    while True:
        team = input(toss_prompt).upper()
        if team in playing_teams:
            return team
        print(f"[ERROR] Invalid team. Toss winner must be either {home_team} or {away_team}")

def get_total_credits(df):
    main_team = df[df['RoleFlag'].notna()]
    total_credits = main_team['Credits'].sum()
    return round(total_credits, 2)

def main():
    try:
        # Define file paths based on environment
        match_data_path = os.path.join(base_dir, 'data', 'MATCH_DATA_COMBINED_DATASET.xlsx')
        squad_data_path = os.path.join(base_dir, 'data', 'SquadPlayerNames_IndianT20League_Dup.xlsx')
        match_metadata_path = os.path.join(base_dir, 'data', 'MATCH_METADATA.xlsx')
        credit_data_path = os.path.join(base_dir, 'data', 'credits_reference_with_priority.xlsx')

        # Initialize TeamSelector
        logging.info("Initializing TeamSelector...")
        team_selector = TeamSelector(
            match_data_path,
            squad_data_path,
            match_metadata_path,
            credit_data_path
        )

        # Get user inputs
        match_id = input("Enter match ID (e.g., 33 for IPL match 33): ")
        home_team = input("Enter Home Team (e.g., DC): ").strip().upper()
        away_team = input("Enter Away Team (e.g., RR): ").strip().upper()
        venue = input("Enter Venue (e.g., Delhi): ").strip()
        toss_winner = input(f"Enter Toss Winner ({home_team}/{away_team}): ").strip().upper()
        toss_decision = input("Enter Toss Decision (Bat/Bowl): ").strip().capitalize()

        # Predict the final team
        logging.info("Predicting the best team...")
        final_team = team_selector.predict(match_id, home_team, away_team, venue, toss_winner, toss_decision)

        # Format the output
        output_df = final_team.copy().reset_index(drop=True)
        output_df['S.No'] = output_df.index + 1
        output_df['RoleFlag'] = 'Player'

        if 'C' in output_df.columns:
            captain_idx = output_df[output_df['C'] == True].index
            if not captain_idx.empty:
                output_df.at[captain_idx[0], 'RoleFlag'] = 'Captain'

        if 'VC' in output_df.columns:
            vice_captain_idx = output_df[output_df['VC'] == True].index
            if not vice_captain_idx.empty:
                output_df.at[vice_captain_idx[0], 'RoleFlag'] = 'Vice-Captain'

        if len(output_df) > 11:
            for idx in range(11, min(len(output_df), 15)):
                output_df.at[idx, 'RoleFlag'] = 'Backup'

        output_df = output_df[['S.No', 'Credits', 'Player Role', 'Player', 'Team', 'RoleFlag']]

        # Save to outputs
        output_path = os.path.join(base_dir, 'outputs', 'final_team_output.csv')
        output_df.to_csv(output_path, index=False)

        logging.info("Final Team (Submission Format):\n%s", output_df.to_string(index=False))

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()
