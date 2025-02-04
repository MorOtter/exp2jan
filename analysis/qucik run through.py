import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
participants_file = "downloads/exp2 participants.csv"
packets_file = "downloads/exp2 packets.csv"
trials_file = "downloads/exp2 trials.csv"

try:
    # Load the dataframes
    participants_df = pd.read_csv(participants_file)
    packets_df = pd.read_csv(packets_file)
    trials_df = pd.read_csv(trials_file)

    # Filter participant IDs with NULL feedback
    invalid_participants = participants_df[participants_df['feedback'].isna()]['participant_id'].unique()

    # Remove packet entries with invalid participant IDs
    valid_trials_df = trials_df[~trials_df['participant_id'].isin(invalid_participants)]

    # Remove rows from packets_df where 'accepted' is TRUE
    packets_df = packets_df[packets_df['accepted'] != True]

    # Merge trials and packets data
    merged_df = valid_trials_df.merge(packets_df, on='trial_id', how='inner')

    # Ensure "accepted" column exists and filter rows where it is True
    if 'accepted' in merged_df.columns:
        merged_df = merged_df[merged_df['accepted'] == False]

    # Merge participants to add "condition"
    merged_df = merged_df.merge(participants_df[['participant_id', 'condition']], on='participant_id', how='left')

    # Handle missing user_input or advisor_recommendation by dropping such rows
    merged_df = merged_df.dropna(subset=['user_input', 'advisor_recommendation'])

    # Add a correctness column based on user_input matching advisor_recommendation
    merged_df['correct'] = np.where(
        merged_df['user_input'].str.strip() == merged_df['advisor_recommendation'].str.strip(), 1, 0
    )

    # Identify participants who completed the maximum trial number
    max_trial_per_participant = merged_df.groupby('participant_id')['trial_number'].max()

    # Keep only participants who completed at least trial 2
    completed_participants = max_trial_per_participant[max_trial_per_participant >= 2].index
    merged_df = merged_df[merged_df['participant_id'].isin(completed_participants)]

    # Check if any data remains after filtering
    if merged_df.empty:
        print("No data available after filtering. Please check the filtering criteria.")
    else:
        # Calculate mean correctness and standard error per trial grouped by condition
        trial_stats = (merged_df.groupby(['trial_number', 'condition'])
                       .agg(mean_correct=('correct', 'mean'),
                            se_correct=('correct', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0))
                       .reset_index())

        # Plotting the graph with error bars
        plt.figure(figsize=(12, 6))
        for condition in trial_stats['condition'].unique():
            subset = trial_stats[trial_stats['condition'] == condition]
            plt.errorbar(subset['trial_number'], subset['mean_correct'], yerr=subset['se_correct'],
                         marker='o', capsize=5, label=condition)

        plt.title('Mean Correct Classification by Trial (with SE)')
        plt.xlabel('Trial Number')
        plt.ylabel('Mean Correct Classification')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Descriptive statistics for correctness by trial and condition
        desc_stats = merged_df.groupby(['trial_number', 'condition']).agg(
            mean=('correct', 'mean'),
            median=('correct', 'median'),
            var=('correct', 'var'),
            count=('correct', 'count'),
            unique_participants=('participant_id', 'nunique')
        ).reset_index()

        # Save descriptive statistics to a CSV file
        output_file = "trial_descriptive_statistics.csv"
        desc_stats.to_csv(output_file, index=False)
        print(f"Descriptive statistics saved to {output_file}")

        print("\nDescriptive Statistics for Correctness by Trial (including unique participant counts):")
        print(desc_stats.to_string(index=False))

except FileNotFoundError as e:
    print(f"Error: {e}. Please check file paths.")
except KeyError as e:
    print(f"Missing column in data: {e}. Ensure all required columns are present.")
except Exception as e:
    print(f"An unexpected error occurred: {e}.")