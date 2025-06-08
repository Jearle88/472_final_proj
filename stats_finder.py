import pandas as pd
import os
import glob
from pathlib import Path


def filter_team_data_by_schema(team_id, data_directory, output_file="team_data_filtered.csv"):
    """
    Filter data from multiple CSV files based on team ID using the known database schema.
    Focuses on relevant tables: teams, teamStats, standings, keyEvents_2024, plays_2024_EN, etc.

    Args:
        team_id (int): The team ID to filter for (e.g., 656)
        data_directory (str): Path to directory containing CSV files
        output_file (str): Name of output CSV file
    """

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_directory}")
        return

    print(f"Found {len(csv_files)} CSV files to process...")

    # Define table priorities and their expected team ID columns these can be changed to what ever stats
    table_mappings = {
        'teams': {'team_col': 'teamId', 'priority': 1},
        'teamStats': {'team_col': 'teamId', 'priority': 2},
        'standings': {'team_col': 'teamId', 'priority': 2},
        'keyEvents': {'team_col': 'teamId', 'priority': 3},
        'plays_2024': {'team_col': 'teamId', 'priority': 3},
        'lineup_2024': {'team_col': 'teamId', 'priority': 4},
        'fixtures': {'team_col': ['homeTeamId', 'awayTeamId'], 'priority': 3},
        'playerStats': {'team_col': 'teamId', 'priority': 4},
        'players': {'team_col': 'teamId', 'priority': 5}
    }

    team_data_collection = {}

    # Process each CSV file
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path).lower()
            print(f"Processing: {filename}")

            # Read the CSV file
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            # Determine table type from filename
            table_type = None
            for table_name in table_mappings.keys():
                if table_name.lower() in filename:
                    table_type = table_name
                    break

            if not table_type:
                print(f"  Unknown table type, checking for team columns...")
                # Fallback: look for any team ID column
                possible_team_cols = ['teamId', 'team_id', 'homeTeamId', 'awayTeamId']
                found_team_col = None
                for col in possible_team_cols:
                    if col in df.columns:
                        found_team_col = col
                        break

                if found_team_col:
                    team_data = df[df[found_team_col] == team_id].copy()
                    if not team_data.empty:
                        team_data['source_file'] = filename
                        team_data['table_type'] = 'unknown'
                        team_data_collection[f'unknown_{filename}'] = team_data
                        print(f"  Found {len(team_data)} records (unknown table type)")
                continue

            # Process known table types
            team_cols = table_mappings[table_type]['team_col']
            if isinstance(team_cols, str):
                team_cols = [team_cols]

            all_team_data = []
            for team_col in team_cols:
                if team_col in df.columns:
                    team_data = df[df[team_col] == team_id].copy()
                    if not team_data.empty:
                        team_data['source_file'] = filename
                        team_data['table_type'] = table_type
                        team_data['team_relation'] = team_col  # Track which column matched
                        all_team_data.append(team_data)
                        print(f"  Found {len(team_data)} records via {team_col}")

            if all_team_data:
                combined_table_data = pd.concat(all_team_data, ignore_index=True)
                team_data_collection[table_type] = combined_table_data
            else:
                print(f"  No data found for team {team_id}")

        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            continue

    if not team_data_collection:
        print(f"\nNo data found for team ID {team_id} in any files.")
        return None

    # Combine all data
    all_data = []
    for table_name, data in team_data_collection.items():
        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True, sort=False)

    # Save complete dataset
    combined_data.to_csv(output_file, index=False)
    print(f"\nSuccess! All team data saved to {output_file}")
    print(f"Total records found: {len(combined_data)}")

    # Create specialized extracts
    extract_team_specific_data(team_data_collection, team_id)

    return team_data_collection


def extract_team_specific_data(team_data_collection, team_id):
    """
    Create focused extracts based on the database schema.
    """
    print(f"\n=== CREATING SPECIALIZED EXTRACTS ===")

    # 1. Team Performance Summary
    performance_data = []

    if 'teamStats' in team_data_collection:
        stats_data = team_data_collection['teamStats']
        performance_data.append(stats_data)
        print(f"Team Stats: {len(stats_data)} records")

    if 'standings' in team_data_collection:
        standings_data = team_data_collection['standings']
        performance_data.append(standings_data)
        print(f"Standings: {len(standings_data)} records")

    if performance_data:
        perf_combined = pd.concat(performance_data, ignore_index=True)
        perf_file = f"team_{team_id}_performance.csv"
        perf_combined.to_csv(perf_file, index=False)
        print(f"Performance data saved to {perf_file}")

    # 2. Events and Actions
    if 'keyEvents' in team_data_collection:
        events_data = team_data_collection['keyEvents']
        events_file = f"team_{team_id}_key_events.csv"
        events_data.to_csv(events_file, index=False)
        print(f"Key Events: {len(events_data)} records saved to {events_file}")

        # Extract event types if available
        if 'keyEventTypeId' in events_data.columns:
            event_types = events_data['keyEventTypeId'].value_counts()
            print("Event Types Found:")
            for event_type, count in event_types.items():
                print(f"  Type {event_type}: {count} events")

    # 3. Plays/Actions
    if 'plays_2024' in team_data_collection:
        plays_data = team_data_collection['plays_2024']
        plays_file = f"team_{team_id}_plays.csv"
        plays_data.to_csv(plays_file, index=False)
        print(f"Plays: {len(plays_data)} records saved to {plays_file}")

        # Show play types if available
        play_columns = ['playType', 'playName', 'eventId']
        available_play_cols = [col for col in play_columns if col in plays_data.columns]
        if available_play_cols:
            print(f"Play data columns: {available_play_cols}")

    # 4. Match Fixtures (home and away)
    if 'fixtures' in team_data_collection:
        fixtures_data = team_data_collection['fixtures']
        fixtures_file = f"team_{team_id}_fixtures.csv"
        fixtures_data.to_csv(fixtures_file, index=False)
        print(f"Fixtures: {len(fixtures_data)} records saved to {fixtures_file}")

        # Show home vs away breakdown
        if 'team_relation' in fixtures_data.columns:
            relation_counts = fixtures_data['team_relation'].value_counts()
            print("Match types:")
            for relation, count in relation_counts.items():
                match_type = "Home" if "home" in relation.lower() else "Away"
                print(f"  {match_type} games: {count}")


def analyze_team_schema(team_id, data_directory):
    """
    Analyze what data is available for a team based on the known schema.
    """
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))

    print(f"Schema-based analysis for team ID {team_id}...")
    print(f"Checking {len(csv_files)} CSV files...\n")

    # Expected tables and what they contain
    expected_tables = {
        'teams': 'Basic team information (name, colors, venue, etc.)',
        'teamStats': 'Team performance statistics',
        'standings': 'League standings and rankings',
        'keyEvents': 'Key events in matches',
        'plays_2024': 'Detailed play-by-play data',
        'fixtures': 'Match fixtures (home/away games)',
        'lineup_2024': 'Team lineups',
        'playerStats': 'Individual player statistics',
        'players': 'Player roster information'
    }

    found_tables = {}

    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path).lower()
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            # Check each expected table
            for table_name, description in expected_tables.items():
                if table_name.lower() in filename:
                    # Look for team data
                    team_cols = ['teamId', 'homeTeamId', 'awayTeamId']
                    team_data_count = 0

                    for col in team_cols:
                        if col in df.columns:
                            team_data_count += len(df[df[col] == team_id])

                    if team_data_count > 0:
                        found_tables[table_name] = {
                            'file': filename,
                            'records': team_data_count,
                            'description': description,
                            'columns': list(df.columns)
                        }
                        print(f"✓ {table_name.upper()}: {team_data_count} records")
                        print(f"  File: {filename}")
                        print(f"  Purpose: {description}")
                        print(
                            f"  Key columns: {[col for col in df.columns if any(keyword in col.lower() for keyword in ['team', 'event', 'play', 'goal', 'shot', 'win', 'loss'])]}")
                        print()
                    break

        except Exception as e:
            continue

    # Show missing tables
    missing_tables = set(expected_tables.keys()) - set(found_tables.keys())
    if missing_tables:
        print("Missing or empty tables:")
        for table in missing_tables:
            print(f"  - {table}: {expected_tables[table]}")

    return found_tables


def get_event_details(team_id, data_directory, event_id=None):
    """
    Get detailed information about specific events or all events for a team.
    """
    print(f"Getting event details for team {team_id}...")

    # Look for key events and plays
    event_files = ['keyEvents', 'plays_2024']
    all_events = []

    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))

    for file_path in csv_files:
        filename = os.path.basename(file_path).lower()

        # Check if this is an events file
        if any(event_type in filename for event_type in event_files):
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()

                # Find team data
                if 'teamId' in df.columns:
                    team_events = df[df['teamId'] == team_id].copy()

                    if event_id and 'eventId' in df.columns:
                        team_events = team_events[team_events['eventId'] == event_id]

                    if not team_events.empty:
                        team_events['source_table'] = filename
                        all_events.append(team_events)

            except Exception as e:
                continue

    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)

        if event_id:
            output_file = f"team_{team_id}_event_{event_id}_details.csv"
            print(f"Event {event_id} details: {len(combined_events)} records")
        else:
            output_file = f"team_{team_id}_all_events_details.csv"
            print(f"All events: {len(combined_events)} records")

        combined_events.to_csv(output_file, index=False)
        print(f"Event details saved to {output_file}")

        # Show event summary
        if 'keyEventTypeId' in combined_events.columns:
            print("\nEvent types:")
            event_summary = combined_events['keyEventTypeId'].value_counts()
            for event_type, count in event_summary.items():
                print(f"  Type {event_type}: {count} occurrences")

        return combined_events
    else:
        print("No event data found")
        return None


# Example usage
if __name__ == "__main__":
    # Configuration
    TEAM_ID = 83 # Change this to your desired team ID
    DATA_DIR = "C:/Users/Johnt/OneDrive/Documents/GitHub/472_final_proj/archive/base_data"  # Change this to your data directory path

    print("=== TEAM DATA EXTRACTION TOOL ===")
    print(f"Target Team ID: {TEAM_ID}")
    print(f"Data Directory: {DATA_DIR}\n")

    # Step 1: Analyze available data
    print("=== SCHEMA ANALYSIS ===")
    found_tables = analyze_team_schema(TEAM_ID, DATA_DIR)

    # Step 2: Extract all team data
    print("\n=== DATA EXTRACTION ===")
    team_data = filter_team_data_by_schema(TEAM_ID, DATA_DIR, f"team_{TEAM_ID}_complete.csv")

    # Step 3: Get event details
    if team_data and ('keyEvents' in team_data or 'plays_2024' in team_data):
        print("\n=== EVENT ANALYSIS ===")
        events = get_event_details(TEAM_ID, DATA_DIR)

        # Example: Get details for a specific event
        # Uncomment the line below and replace with actual event ID
        # specific_event = get_event_details(TEAM_ID, DATA_DIR, event_id=12345)

    print(f"\n=== SUMMARY ===")
    print(f"Team {TEAM_ID} data extraction complete!")
    print("Generated files:")
    print(f"  - team_{TEAM_ID}_complete.csv (all data)")
    print(f"  - team_{TEAM_ID}_performance.csv (stats & standings)")
    print(f"  - team_{TEAM_ID}_key_events.csv (events)")
    print(f"  - team_{TEAM_ID}_plays.csv (plays)")
    print(f"  - team_{TEAM_ID}_fixtures.csv (matches)")