import re
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from nba_api.stats.endpoints import LeagueGameLog, BoxScoreTraditionalV2, leaguedashteamstats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

data = """
CHI
CHI
DET
DET
Confirmed Lineup
PG Josh Giddey
SG Lonzo Ball
SF Coby White
PF Ayo Dosunmu
C N. Vucevic
MAY NOT PLAY
G T. Horton-Tucker GTD
C J. Smith GTD
F D. Terry GTD
F T. Craig OUT
Confirmed Lineup
PG C. Cunningham
SG Tim Hardaway
SF A. Thompson
PF Tobias Harris
C Jalen Duren
MAY NOT PLAY
G Jaden Ivey OUT

Make The Right Picks

If you play on Pick6, you should check out the RotoWire Picks tool. Get our best picks for today's NBA games.
See Our Top Picks For Pick6
3:30 PM ET
Tickets
alert Alerts
DAL
DAL
CLE
CLE
Confirmed Lineup
PG S. Dinwiddie
SG Dante Exum
SF Klay Thompson
PF O. Prosper
C Kylor Kelley
MAY NOT PLAY
G M. Christie GTD
C D. Gafford GTD
G K. Irving GTD
F P. Washington GTD
C A. Davis OUT
C D. Lively OUT
C D. Powell OUT
Confirmed Lineup
PG D. Garland
SG D. Mitchell
SF Max Strus
PF Evan Mobley
C Jarrett Allen
MAY NOT PLAY
G C. Porter GTD
F I. Okoro OUT
F L. Travers OUT
F Dean Wade OUT
3:30 PM ET
Tickets
alert Alerts
LAC
LAC
TOR
TOR
Confirmed Lineup
PG James Harden
SG Amir Coffey
SF Kawhi Leonard
PF Derrick Jones
C Ivica Zubac
MAY NOT PLAY
G C. Christie OUT
G Kris Dunn OUT
C D. Eubanks OUT
G P. Mills OUT
G N. Powell OUT
Confirmed Lineup
PG I. Quickley GTD
SG Gradey Dick
SF RJ Barrett
PF S. Barnes
C Jakob Poeltl
MAY NOT PLAY
G D. Mitchell GTD
G I. Quickley GTD
G J. Shead OUT
6:00 PM ET
Tickets
alert Alerts
BOS
BOS
PHI
PHI
Confirmed Lineup
PG Derrick White
SG Jrue Holiday
SF Jaylen Brown
PF Jayson Tatum
C K. Porzingis
MAY NOT PLAY
G P. Pritchard GTD
Confirmed Lineup
PG Tyrese Maxey
SG Kelly Oubre
SF Ricky Council
PF J. Edwards
C G. Yabusele
MAY NOT PLAY
C A. Drummond GTD
G E. Gordon GTD
F C. Martin GTD
C J. Embiid OUT
F P. George OUT
F KJ Martin OUT
8:30 PM ET
Tickets
alert Alerts
MEM
MEM
MIL
MIL
Confirmed Lineup
PG Luke Kennard
SG Desmond Bane
SF Jaylen Wells
PF Jaren Jackson
C Zach Edey
MAY NOT PLAY
G Ja Morant OUT
G M. Smart OUT
G C. Spencer OUT
F V. Williams OUT
Confirmed Lineup
PG D. Lillard
SG Andre Jackson
SF T. Prince
PF G. Antetokounmpo
C Brook Lopez
MAY NOT PLAY
F B. Portis OUT
C L. Robbins OUT
"""

# Team mapping dictionary
TEAM_MAP = {
    'ATL': 'Atlanta Hawks', 
    'BKN': 'Brooklyn Nets', 
    'BOS': 'Boston Celtics', 
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls', 
    'CLE': 'Cleveland Cavaliers', 
    'DAL': 'Dallas Mavericks', 
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons', 
    'GSW': 'Golden State Warriors', 
    'HOU': 'Houston Rockets', 
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers', 
    'LAL': 'Los Angeles Lakers', 
    'MEM': 'Memphis Grizzlies', 
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks', 
    'MIN': 'Minnesota Timberwolves', 
    'NOP': 'New Orleans Pelicans', 
    'NYK': 'New York Knicks',
    'ORL': 'Orlando Magic', 
    'PHI': 'Philadelphia 76ers', 
    'PHX': 'Phoenix Suns', 
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings', 
    'SAS': 'San Antonio Spurs', 
    'TOR': 'Toronto Raptors', 
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards', 
    'OKC': 'Oklahoma City Thunder'
}

# File paths
UPCOMING_FILE = r'D:\SportsBetting\2025\basketball\ripoff\upcoming.csv'

def parse_lineup_status():
    """Parse player lineup data and return players listed as OUT"""
    # In this example, assume 'data' is a global string (e.g. read from a file)
    lines = data.split("\n")
    out_players = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        elif re.match(r"(PG|SG|SF|PF|C|G|F)", line):
            if len(line.split()) > 3 and line.split()[3] == 'OUT':
                out_players.append(f'{line.split()[1][:1]}{line.split()[2]}')
    return out_players

def load_team_elo(games):
    data = games.copy()
    data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'], format='%Y-%m-%d')
    one_week_ago = pd.Timestamp.today() - timedelta(days=7)
    data = data[data['GAME_DATE'] >= one_week_ago]

    initial_elo = 1500
    teams = set(data["TEAM_ABBREVIATION"].unique())
    elo_ratings = {team: initial_elo for team in teams}

    K = 200

    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_elo(winner_elo, loser_elo, K=K):
        expected_winner = expected_score(winner_elo, loser_elo)
        expected_loser = expected_score(loser_elo, winner_elo)
        
        new_winner_elo = winner_elo + K * (1 - expected_winner)
        new_loser_elo = loser_elo + K * (0 - expected_loser)
        
        return new_winner_elo, new_loser_elo

    for _, row in data.iterrows():
        matchup = row["MATCHUP"]
        if " vs. " in matchup:
            team_a, team_b = matchup.split(" vs. ")
        elif " @ " in matchup:
            team_a, team_b = matchup.split(" @ ")
        else:
            raise ValueError(f"Unknown matchup format: {matchup}")        
        
        wl = row["WL"]
        
        if wl == "W":
            winner = row["TEAM_ABBREVIATION"]
            loser = team_b if row["TEAM_ABBREVIATION"] == team_a else team_a
        else:
            loser = row["TEAM_ABBREVIATION"]
            winner = team_b if row["TEAM_ABBREVIATION"] == team_a else team_a
        
        winner_elo = elo_ratings[winner]
        loser_elo = elo_ratings[loser]
        
        new_winner_elo, new_loser_elo = update_elo(winner_elo, loser_elo)
        elo_ratings[winner] = new_winner_elo
        elo_ratings[loser] = new_loser_elo

    return elo_ratings

def elo_win_probability(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def fetch_data(days: int):
    """Fetch game and player data for the last N days."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    try:
        game_log = LeagueGameLog(
            season='2024-25',
            date_from_nullable=start_date.strftime('%Y-%m-%d'),
            date_to_nullable=end_date.strftime('%Y-%m-%d')
        )
        games = game_log.get_data_frames()[0]

        all_player_stats = []
        for game_id in tqdm(games['GAME_ID'].unique(), desc="Fetching player stats"):
            box_score = BoxScoreTraditionalV2(game_id=game_id)
            player_stats = box_score.get_data_frames()[0]
            all_player_stats.append(player_stats)
            time.sleep(0.5)  # Rate limit

        return games, pd.concat(all_player_stats, ignore_index=True)
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise

def get_team_def_ratings(season: str = "2024-25"):
    """Fetch defensive ratings for all NBA teams."""
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
        df_def = team_stats.get_data_frames()[0]
        df_def['DEF_RATING'] = 100 * (1 - (df_def['STL'] + df_def['BLK'] + df_def['DREB']) / df_def['FGA'])
        return df_def[['TEAM_ID', 'TEAM_NAME', 'DEF_RATING']]
    except Exception as e:
        logging.error(f"Error fetching defensive ratings: {e}")
        raise


def predict_next_game(model, minutes: float, fga_avg_5: float, fg_pct_avg_5: float, def_rating: float):
    """Predict player points for the next game based on inputs."""
    input_data = np.array([[minutes, fga_avg_5, fg_pct_avg_5, def_rating]])
    return model.predict(input_data)[0]

def home(matchup):
    if " vs. " in matchup:
        return matchup.split(" vs. ")[0]
    elif " @ " in matchup:
        return matchup.split(" @ ")[1]
    else:
        raise ValueError(f"Unknown matchup format: {matchup}")
    
def visitor(matchup):
    if " vs. " in matchup:
        return matchup.split(" vs. ")[1]
    elif " @ " in matchup:
        return matchup.split(" @ ")[0]
    else:
        raise ValueError(f"Unknown matchup format: {matchup}")

def get_team_def_ratings(season: str = "2024-25"):
    """Fetch defensive ratings for all NBA teams."""
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
        df_def = team_stats.get_data_frames()[0]
        df_def['DEF_RATING'] = 100 * (1 - (df_def['STL'] + df_def['BLK'] + df_def['DREB']) / df_def['FGA'])
        return df_def[['TEAM_ID', 'TEAM_NAME', 'DEF_RATING']]
    except Exception as e:
        logging.error(f"Error fetching defensive ratings: {e}")
        raise

games, all_player_stats = fetch_data(days=60)

all_player_stats['scratch'] = all_player_stats.apply(lambda x: f"{x['PLAYER_NAME'].split()[0][0]}{x['PLAYER_NAME'].split()[1]}" not in parse_lineup_status(), axis=1)
all_player_stats = all_player_stats[all_player_stats['scratch'] == True].copy()

df = all_player_stats.merge(games[['GAME_ID', 'GAME_DATE', 'MATCHUP']], on='GAME_ID', how='left')
df['TEAM'] = df['TEAM_ABBREVIATION'].map(TEAM_MAP)
df['HOME'] = df['MATCHUP'].apply(home).map(TEAM_MAP)
df['VISITOR'] = df['MATCHUP'].apply(visitor).map(TEAM_MAP)
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])
df['MIN'] = df['MIN'].str.split(':').str[0].astype(float)
df.replace(-np.inf, np.nan, inplace=True)
df = df.fillna(0)

defence = get_team_def_ratings()
df = df.merge(defence, left_on='TEAM', right_on='TEAM_NAME', how='left')

for stat in ['FGA', 'FG_PCT', 'FG3M', 'FG3_PCT', 'DEF_RATING']: df[f'{stat}_AVG_5'] = df.groupby('PLAYER_ID')[stat].transform(lambda x: x.rolling(5, min_periods=1).mean())

features = ['MIN', 'FGA_AVG_5', 'FG_PCT_AVG_5', 'DEF_RATING_AVG_5']
X = df[features]
y = df['PTS']

model = RandomForestRegressor()
model.fit(X, y)

df['Predicted'] = 0

df_latest = df.sort_values(by='GAME_DATE').groupby('PLAYER_ID').tail(1).copy()
import numpy as np

def average_prediction(row, runs=100):
    predictions = [
        predict_next_game(
            model, 
            48, 
            row['FGA_AVG_5'], 
            row['FG_PCT_AVG_5'], 
            row['DEF_RATING_AVG_5']
        )
        for _ in range(runs)
    ]
    return np.mean(predictions)

df_latest['Predicted'] = df_latest.apply(average_prediction, axis=1)

df_latest = df_latest.sort_values(by='Predicted').groupby('TEAM_ABBREVIATION').head(5)
df_latest['Predicted'] = df_latest['Predicted'] * 2

matchups = pd.read_csv(UPCOMING_FILE)
matchups['Date'] = pd.to_datetime(matchups['Date'], format='%a %b %d %Y')
matchups['Date'] = matchups['Date'].dt.strftime('%Y-%m-%d')
matchups = matchups[matchups['Date'] == datetime.now().strftime('%Y-%m-%d')]

matchups = matchups[['Home', 'Visitor']].drop_duplicates().values
unique_games = []
for team1, team2 in matchups:
    if f"{team2} vs {team1}" not in unique_games:
        unique_games.append(f"{team1} vs {team2}")

for game in unique_games:
    teams = game.split(' vs ')
    if teams[0] != 'nan':
        team1_score = df_latest[df_latest['TEAM'] == teams[0]]['Predicted'].sum()
        team2_score = df_latest[df_latest['TEAM'] == teams[1]]['Predicted'].sum()
        print(f"\nScores for {teams[0]} and {teams[1]}:")
        print(f"{teams[0]}: {round(team1_score, 2)}")
        print(f"{teams[1]}: {round(team2_score, 2)}")
