import pandas as pd
import os
import glob
from pathlib import Path


def filter_team_data_by_schema(team_id, data_directory, output_file="team_data_filtered.csv"):
    """
    Filter data from multiple CSV files based on team ID by dynamically detecting team columns.
    Reads column names from CSV files instead of using hardcoded mappings.

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

    team_data_collection = {}

    # Process each CSV file
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path).lower()
            print(f"Processing: {filename}")

            # Read the CSV file
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            # Dynamically find team ID columns by checking column names
            possible_team_cols = []
            for col in df.columns:
                col_lower = col.lower()
                # Look for columns that likely contain team IDs
                if any(pattern in col_lower for pattern in ['teamid', 'team_id', 'hometeamid', 'awayteamid']):
                    possible_team_cols.append(col)

            if not possible_team_cols:
                print(f"  No team ID columns found in {filename}")
                continue

            # Determine table type from filename (remove file extension and special characters)
            table_type = os.path.splitext(filename)[0].replace('_', '').replace('-', '').lower()

            print(f"  Found team columns: {possible_team_cols}")

            all_team_data = []
            for team_col in possible_team_cols:
                if team_col in df.columns:
                    # Check if this column contains the team_id we're looking for
                    team_data = df[df[team_col] == team_id].copy()
                    if not team_data.empty:
                        team_data['source_file'] = filename
                        team_data['table_type'] = table_type
                        team_data['team_relation'] = team_col  # Track which column matched
                        all_team_data.append(team_data)
                        print(f"  Found {len(team_data)} records via {team_col}")

            if all_team_data:
                combined_table_data = pd.concat(all_team_data, ignore_index=True)
                # Use filename as key if multiple files have same table type
                table_key = f"{table_type}_{filename}" if table_type in team_data_collection else table_type
                team_data_collection[table_key] = combined_table_data
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

    return team_data_collection


def filter_multiple_teams_data(team_ids, data_directory, consolidated_output="all_teams_consolidated.csv"):
    """
    Filter data for multiple teams and create one consolidated output file.

    Args:
        team_ids (list): List of team IDs to filter for
        data_directory (str): Path to directory containing CSV files
        consolidated_output (str): Name of consolidated output CSV file
    """
    print(f"=== CONSOLIDATING DATA FOR TEAMS: {team_ids} ===")

    all_teams_data = []

    for team_id in team_ids:
        print(f"\n--- Processing Team ID: {team_id} ---")
        team_data_collection = filter_team_data_by_schema(
            team_id,
            data_directory,
            f"temp_team_{team_id}_complete.csv"
        )

        if team_data_collection:
            # Combine all data for this team
            team_all_data = []
            for table_name, data in team_data_collection.items():
                # Add team_id column to ensure we can identify which team each record belongs to
                data_with_team = data.copy()
                data_with_team['processed_team_id'] = team_id
                team_all_data.append(data_with_team)

            if team_all_data:
                team_combined = pd.concat(team_all_data, ignore_index=True, sort=False)
                all_teams_data.append(team_combined)
                print(f"Team {team_id}: {len(team_combined)} total records collected")

        # Clean up temporary file
        temp_file = f"temp_team_{team_id}_complete.csv"
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if all_teams_data:
        # Consolidate all teams into one DataFrame
        consolidated_df = pd.concat(all_teams_data, ignore_index=True, sort=False)

        # Save consolidated data
        consolidated_df.to_csv(consolidated_output, index=False)

        print(f"\n=== CONSOLIDATION COMPLETE ===")
        print(f"Total records across all teams: {len(consolidated_df)}")
        print(f"Consolidated data saved to: {consolidated_output}")

        # Show breakdown by team
        if 'processed_team_id' in consolidated_df.columns:
            team_breakdown = consolidated_df['processed_team_id'].value_counts().sort_index()
            print("\nRecords per team:")
            for team_id, count in team_breakdown.items():
                print(f"  Team {team_id}: {count} records")

        # Show breakdown by table type
        if 'table_type' in consolidated_df.columns:
            table_breakdown = consolidated_df['table_type'].value_counts()
            print("\nRecords by table type:")
            for table_type, count in table_breakdown.items():
                print(f"  {table_type}: {count} records")

        return consolidated_df
    else:
        print("No data found for any of the specified teams.")
        return None


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
    Analyze what data is available for a team by dynamically detecting columns.
    """
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))

    print(f"Dynamic schema analysis for team ID {team_id}...")
    print(f"Checking {len(csv_files)} CSV files...\n")

    found_tables = {}

    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path).lower()
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            # Dynamically find team ID columns
            team_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in ['teamid', 'team_id', 'hometeamid', 'awayteamid']):
                    team_cols.append(col)

            if team_cols:
                team_data_count = 0
                for col in team_cols:
                    if col in df.columns:
                        team_data_count += len(df[df[col] == team_id])

                if team_data_count > 0:
                    table_name = os.path.splitext(filename)[0]
                    found_tables[table_name] = {
                        'file': filename,
                        'records': team_data_count,
                        'team_columns': team_cols,
                        'all_columns': list(df.columns)
                    }
                    print(f"✓ {table_name.upper()}: {team_data_count} records")
                    print(f"  File: {filename}")
                    print(f"  Team columns found: {team_cols}")
                    print(f"  Total columns: {len(df.columns)}")
                    print()

        except Exception as e:
            continue

    if not found_tables:
        print("No tables with team data found.")

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

def load_team_ids_from_csv(file_path, column_name):
    """
    Load all unique team IDs from a given column in a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        column_name (str): Name of the column containing team IDs.

    Returns:
        list: A list of unique team IDs as integers.
    """
    df = pd.read_csv(file_path)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    team_ids = df[column_name].dropna().astype(int).unique().tolist()
    return team_ids
# Example usage
if __name__ == "__main__":
    # Configuration
     # Change this to your desired team IDs
    DATA_DIR = "C:/Users/Johnt/OneDrive/Documents/GitHub/472_final_proj/test_data"  # Change this to your data directory path
    TEAM_IDs= load_team_ids_from_csv( "C:/Users/Johnt/OneDrive/Documents/GitHub/472_final_proj/archive/base_data/teams.csv", "teamId")
    print("=== TEAM DATA EXTRACTION TOOL ===")
    print(f"Target Team IDs: {TEAM_IDs}")
    print(f"Data Directory: {DATA_DIR}\n")

    # NEW: Create consolidated output for all teams
    consolidated_data = filter_multiple_teams_data(
        TEAM_IDs,
        DATA_DIR,
        "all_teams_consolidated_data.csv"
    )

    """
    # Optional: Still create individual team analysis if needed
    print(f"\n=== INDIVIDUAL TEAM ANALYSIS (Optional) ===")
    for TEAM_ID in TEAM_IDs:
        print(f"\n--- Analyzing Team {TEAM_ID} ---")

        # Step 1: Analyze available data
        found_tables = analyze_team_schema(TEAM_ID, DATA_DIR)

        # Step 2: Extract all team data (individual files)
        team_data = filter_team_data_by_schema(TEAM_ID, DATA_DIR, f"team_{TEAM_ID}_complete.csv")

        # Step 3: Get event details
        if team_data and ('keyEvents' in team_data or 'plays_2024' in team_data):
            events = get_event_details(TEAM_ID, DATA_DIR)
   
    print(f"\n=== FINAL SUMMARY ===")
    print("Generated files:")
    print("  - all_teams_consolidated_data.csv (ALL TEAMS COMBINED)")
    for TEAM_ID in TEAM_IDs:
        print(f"  - team_{TEAM_ID}_complete.csv (individual team data)")
        print(f"  - team_{TEAM_ID}_performance.csv (stats & standings)")
        print(f"  - team_{TEAM_ID}_key_events.csv (events)")
        print(f"  - team_{TEAM_ID}_plays.csv (plays)")
        print(f"  - team_{TEAM_ID}_fixtures.csv (matches)")
    """