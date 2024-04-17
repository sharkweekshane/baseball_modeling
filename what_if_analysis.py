### Import data

import pandas as pd
import os

def load_and_clean_data(filename, columns, rename_map):
    df = pd.read_csv(os.path.join(PATH, filename))
    cleaned_df = df[columns].copy()
    cleaned_df.columns = rename_map
    cleaned_df['Name'] = cleaned_df['Name'].str.replace('[^a-zA-Z ]', '', regex=True)
    return cleaned_df

# Define the path and load the data
PATH = '/Users/shane/Documents/data_science/what_if/'
ted_stats = load_and_clean_data('ted_stats.csv', ['Season', 'Name', 'Age', 'R', 'HR', 'RBI', 'SB', 'AVG', 'OBP', 'wRC+', 'WAR'], 
                                                 ['Season', 'Name', 'Age', 'R', 'HR', 'RBI', 'SB', 'AVG', 'OBP', 'wRC+', 'WAR'])
joe_stats = load_and_clean_data('joe_stats.csv', ['Season','Name', 'Age', 'R', 'HR', 'RBI', 'SB', 'AVG', 'OBP', 'wRC+', 'WAR'], 
                                                 ['Season', 'Name', 'Age', 'R', 'HR', 'RBI', 'SB', 'AVG', 'OBP', 'wRC+', 'WAR'])

#%%

### Boxplots of wRC+ w/ Mann-Whitney U test

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import warnings

# Or to ignore all warnings
warnings.filterwarnings("ignore")

# Combine the data into a single DataFrame
combined_stats = pd.concat([ted_stats, joe_stats])
combined_stats['Name'] = combined_stats['Name'].astype('category')  # Ensure the player column is categorical

# Calculate the p-value using the Mann-Whitney U test
stat, p_value = mannwhitneyu(combined_stats[combined_stats['Name'] == 'Ted Williams']['wRC+'],
                             combined_stats[combined_stats['Name'] == 'Joe DiMaggio']['wRC+'])

# Set up the plotting environment
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")  # Set the style of the plot

# Create a boxplot to compare Ted Williams and Joe DiMaggio's wRC+
ax = sns.boxplot(x='Name', y='wRC+', data=combined_stats, palette=["salmon", "darkseagreen"])

# Despine and enhance the plot
sns.despine(offset=10, trim=True)  # Clean up the plot edges
plt.title('Comparison of wRC+ Between Ted Williams and Joe DiMaggio', fontsize=16)
plt.xlabel('Player', fontsize=14)
plt.ylabel('wRC+', fontsize=14)

# Function to add significance stars and p-value
def add_significance_star(p, ax, x1, x2, y, h):
    """Add significance stars and p-value based on p-values."""
    # Determine the significance label based on p-value
    if p < 0.001:
        significance_label = '***'  # highly significant
    elif p < 0.01:
        significance_label = '**'  # very significant
    elif p < 0.05:
        significance_label = '*'  # significant
    else:
        significance_label = 'ns'  # not significant

    # Draw line and place text
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    # Place the significance label (e.g., ***) and the p-value
    ax.text((x1+x2)*.5, y+h, significance_label, ha='center', va='bottom', color='black')
    # Place the p-value just below the significance label
    ax.text((x1+x2)*.5, y+h - 4, f'Mann-Whitney U test, p={p:.6f}', ha='center', va='top', color='black')

# Adding the significance star above the groups
max_wrc_plus = combined_stats['wRC+'].max()
height = max_wrc_plus * 0.05  # Line height 5% of the max value
add_significance_star(p_value, ax, 0, 1, max_wrc_plus + height, height)

# Turn off the grid
plt.grid(False)

# Show the plot
plt.show()


#%%

### wRC+ combined scatter

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the plotting environment
plt.figure(figsize=(10, 6))  # Set the figure size
sns.set(style="white")  # Set the style of the plot

# Create scatter plots for both Ted Williams and Joe DiMaggio
sns.scatterplot(x='Age', y='wRC+', data=ted_stats, color='darkseagreen', s=100, marker='o', label='Ted Williams')
sns.scatterplot(x='Age', y='wRC+', data=joe_stats, color='salmon', s=100, marker='o', label='Joe DiMaggio')

# Despine and remove grid lines
sns.despine(left=True, bottom=True)
plt.grid(False)

# Enhance the plot with labels, legend, and title
plt.title('Ted Williams & Joe DiMaggio wRC+ vs Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('wRC+', fontsize=14)
plt.legend(title='Player', title_fontsize='13', fontsize='12')

# Ensure x-axis is a whole number
plt.xticks(range(int(min(ted_stats['Age'].min(), joe_stats['Age'].min())), int(max(ted_stats['Age'].max(), joe_stats['Age'].max())) + 1))

#%%

### wRC+ with war highlights

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set up the plotting environment
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharey=True)  # Two subplots, sharing Y axis
sns.set(style="white")  # Set the style of the plot

# Create scatter plot for Ted Williams without a legend
sns.scatterplot(ax=axes[0], x='Age', y='wRC+', data=ted_stats, color='darkseagreen', s=300, marker='o')
axes[0].set_title('Ted Williams wRC+ vs Age', fontsize=25)
axes[0].set_xlabel('Age', fontsize=18)
axes[0].set_ylabel('wRC+', fontsize=18)

# Highlight ages 24, 25, 26 for Ted Williams
axes[0].axvspan(24, 26, color='goldenrod', alpha=0.2)
# axes[0].axvspan(33, 34, color='goldenrod', alpha=0.2)

# Create scatter plot for Joe DiMaggio without a legend
sns.scatterplot(ax=axes[1], x='Age', y='wRC+', data=joe_stats, color='salmon', s=300, marker='o')
axes[1].set_title('Joe DiMaggio wRC+ vs Age', fontsize=25)
axes[1].set_xlabel('Age', fontsize=18)
axes[1].set_ylabel('wRC+', fontsize=18)

# Highlight ages 28, 29, 30 for Joe DiMaggio
axes[1].axvspan(28, 30, color='goldenrod', alpha=0.2)

# Adjust the plot aesthetics
sns.despine(left=True, bottom=True)  # Despine for cleaner aesthetics
plt.grid(False)  # Remove the grid lines

# Ensure x-axis is a whole number and adjust y-axis limits for alignment
all_ages = pd.concat([ted_stats['Age'], joe_stats['Age']])
all_wrc = pd.concat([ted_stats['wRC+'], joe_stats['wRC+']])
plt.setp(axes, xticks=range(int(all_ages.min()), int(all_ages.max()) + 1))
plt.setp(axes, ylim=(all_wrc.min() - 10, all_wrc.max() + 10))

# Display the plot
plt.tight_layout()
plt.show()
#%%

### Plotting the aging curve with polynomials.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example data points
x_points = np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
y_points = np.array([-29.6, -15.2, -9.6, -3.3, -1.6, -0.2, 0, -0.5, -2.5, -5.8, -8.1, -10.7, -14.1, -19.0, -24.1, -32.1, -40.7, -46.4, -58.1, -59.5])

# Fit a cubic polynomial to the points
degree = 3
coefficients = np.polyfit(x_points, y_points, degree)
polynomial = np.poly1d(coefficients)

# Generate a range of x values for plotting the fitted polynomial
x_range = np.linspace(x_points.min(), x_points.max(), 100)
y_range = polynomial(x_range)

# Set the style for the plot
sns.set_style("whitegrid")

# Plot the actual data points
plt.scatter(x_points, y_points, color='salmon', label='Data Points')

# Plot the fitted polynomial curve
plt.plot(x_range, y_range, color='darkseagreen', label='Fitted Polynomial Curve')

# Customize the plot
plt.title('Aging Curve Fitted to Data Points')
plt.xlabel('Age')
plt.ylabel('Performance (LWTS per 500 PA)')
plt.legend()
plt.grid(False)

# Set x-axis to show each age value
plt.xticks(x_points)

# Add the equation of the polynomial to the plot
eq_text = '{:.4f}x^3 + {:.4f}x^2 + {:.4f}x + {:.4f}'.format(*coefficients)
plt.text(15, min(y_points) - 16, eq_text, fontsize=10, color='purple')
sns.despine()
plt.show()

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import UnivariateSpline
import seaborn as sns

stats = ['R', 'HR', 'RBI', 'SB', 'AVG', 'OBP', 'wRC+', 'WAR']
ted_ages_to_model = [24, 25, 26]
joe_ages_to_model = [28, 29, 30]

def simulate_data(player_stats, ages_to_model, stats, exclude_age=None, exclude_stats=None):
    simulated_data = pd.DataFrame(index=ages_to_model)
    for stat in stats:
        if exclude_age and exclude_stats and stat in exclude_stats:
            # Exclude specific age for specified stats
            data_to_fit = player_stats[player_stats['Age'] != exclude_age]
        else:
            data_to_fit = player_stats
        
        spline_model = UnivariateSpline(data_to_fit['Age'], data_to_fit[stat], s=1)
        predicted_values = spline_model(ages_to_model)
        
        # Format the predicted values according to the stat type
        if stat in ['R', 'HR', 'RBI', 'SB', 'wRC+']:
            simulated_data[stat] = np.round(predicted_values).astype(int)  # Round to nearest integer
        elif stat in ['AVG', 'OBP']:
            simulated_data[stat] = np.round(predicted_values, 3)  # Round to three decimal places
        elif stat == 'WAR':
            simulated_data[stat] = np.round(predicted_values, 1)  # Round to one decimal place
        else:
            simulated_data[stat] = predicted_values  # Leave as is for any other stat
        
    simulated_data['Age'] = ages_to_model  # Set the age column correctly
    return simulated_data.reset_index(drop=True)


# Example usage for Joe DiMaggio excluding age 27 for wRC+ and WAR
ted_simulated = simulate_data(ted_stats, ted_ages_to_model, stats)
joe_simulated = simulate_data(joe_stats, joe_ages_to_model, stats, exclude_age=27, exclude_stats=['wRC+', 'WAR'])

def plot_real_and_simulated_data(real_data, simulated_data, color_real, color_sim, figure_title, y_ranges):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 15), gridspec_kw={'top': 0.92})
    axes = axes.flatten()
    age_ticks = sorted(set(real_data['Age'].unique()).union(simulated_data['Age'].unique()))
    
    for i, stat in enumerate(stats):
        axes[i].scatter(real_data['Age'], real_data[stat], alpha=0.7, c=color_real, s=200, label='Real Data')
        axes[i].scatter(simulated_data['Age'], simulated_data[stat], alpha=0.7, c=color_sim, s=200, label='Simulated Data')
        axes[i].set_ylabel(stat, fontsize=22)
        axes[i].xaxis.set_major_locator(ticker.FixedLocator(age_ticks))
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        
        # Set the range for the y-axis
        if stat in y_ranges:
            min_val, max_val, interval = y_ranges[stat]
            axes[i].set_ylim(min_val, max_val)
            axes[i].set_yticks(np.arange(min_val, max_val + interval, interval))
        
        if i >= 6:
            axes[i].set_xlabel('Age', fontsize=22)
        axes[i].grid(False)
    
    plt.tight_layout()
    fig.suptitle(figure_title, fontsize=30, verticalalignment='top')
    sns.despine()
    plt.show()

# Manually defined y-axis ranges and intervals for each stat
y_ranges = {
    'R': (0, 160, 25),
    'HR': (0, 50, 5),
    'RBI': (0, 180, 25),
    'SB': (0, 10, 2),
    'AVG': (0.2, 0.45, 0.05),
    'OBP': (0.3, 0.6, 0.05),
    'wRC+': (0, 260, 30),
    'WAR': (0, 13, 2)
}

# Example call to the plotting function with y-axis ranges
plot_real_and_simulated_data(ted_stats, ted_simulated, 'darkseagreen', 'goldenrod', "Ted Williams' Career with Simulated Stats", y_ranges)
plot_real_and_simulated_data(joe_stats, joe_simulated, 'salmon', 'goldenrod', "Joe DiMaggio's Career with Simulated Stats", y_ranges)

# Set up simulated + real dataframes
ted_combined = pd.concat([ted_stats, ted_simulated], axis=0)
ted_combined['Name'] = ted_combined['Name'].fillna('Ted Williams')
ted_combined.loc[ted_combined['Age'] == 24, 'Season'] = 1943
ted_combined.loc[ted_combined['Age'] == 25, 'Season'] = 1944
ted_combined.loc[ted_combined['Age'] == 26, 'Season'] = 1945
joe_combined = pd.concat([joe_stats, joe_simulated], axis=0)
joe_combined['Name'] = joe_combined['Name'].fillna('Joe DiMaggio')
joe_combined.loc[joe_combined['Age'] == 28, 'Season'] = 1943
joe_combined.loc[joe_combined['Age'] == 29, 'Season'] = 1944
joe_combined.loc[joe_combined['Age'] == 30, 'Season'] = 1945


# Specify the file path where you want to save the CSV
file_path_ted = '/Users/shane/Documents/data_science/what_if/Ted_Williams_Simulated_Stats_1943-1945.csv'
file_path_joe = '/Users/shane/Documents/data_science/what_if/Joe_DiMaggio_Simulated_Stats_1943-1945.csv'

ted_simulated.to_csv(file_path_ted, index=False)
joe_simulated.to_csv(file_path_joe, index=False)

file_path_ted2 = '/Users/shane/Documents/data_science/what_if/Ted_Williams_Career_with_Simulated.csv'
file_path_joe2 = '/Users/shane/Documents/data_science/what_if/Joe_DiMaggio_Career_with_Simulated.csv'

ted_combined.to_csv(file_path_ted2, index=False)
joe_combined.to_csv(file_path_joe2, index=False)

#%%

import pandas as pd
from fuzzywuzzy import process
import os

def load_and_clean_data(filename, columns, rename_map):
    df = pd.read_csv(os.path.join(PATH, filename))
    cleaned_df = df[columns].copy()
    cleaned_df.columns = rename_map
    cleaned_df['Name'] = cleaned_df['Name'].str.replace('[^a-zA-Z ]', '', regex=True)
    return cleaned_df

def display_leaderboard_changes(player, stat):
    df = dataframes[stat]
    simulated_data = df_ted_sim if player == 'Ted Williams' else df_joe_sim

    df.sort_values(by=stat, ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Rank'] = range(1, len(df) + 1)

    idx = df[df['Name'] == matches[player]].index[0]
    pre_leaderboard = df.iloc[max(0, idx - 5): min(idx + 6, len(df))].copy()

    update_and_get_leaderboard(df, matches[player], simulated_data[stat].sum(), stat)

    post_idx = df[df['Name'] == matches[player]].index[0]
    post_leaderboard = df.iloc[max(0, post_idx - 5): min(post_idx + 6, len(df))]

    print(f"\n--- {player} {stat} Change ---")
    print("Pre-update leaderboard:")
    print(pre_leaderboard.to_string(index=False))
    print()
    print("Post-update leaderboard:")
    print(post_leaderboard.to_string(index=False))

PATH = '/Users/shane/Documents/data_science/what_if/'

# Data loading
dataframes = {
    'R': load_and_clean_data('R.csv', ['#', 'Name', 'R'], ['Rank', 'Name', 'R']),
    'HR': load_and_clean_data('HR.csv', ['#', 'Name', 'HR'], ['Rank', 'Name', 'HR']),
    'RBI': load_and_clean_data('RBI.csv', ['#', 'Name', 'RBI'], ['Rank', 'Name', 'RBI']),
    'WAR': load_and_clean_data('WAR.csv', ['#', 'Name', 'WAR'], ['Rank', 'Name', 'WAR'])
}

# Simulated data
df_ted_sim = pd.read_csv(os.path.join(PATH, 'Ted_Williams_Simulated_Stats_1943-1945.csv'))
df_joe_sim = pd.read_csv(os.path.join(PATH, 'Joe_DiMaggio_Simulated_Stats_1943-1945.csv'))

# Fuzzy matches for player names
matches = {name: process.extractOne(name, dataframes['WAR']['Name'])[0] for name in ['Joe DiMaggio', 'Ted Williams']}

def update_and_get_leaderboard(df, player_name, additional_stat, stat):
    # Make sure the stat being accessed is numeric
    df[stat] = pd.to_numeric(df[stat], errors='coerce')
    # Retrieve the original statistic value, ensure it's numeric, and update it
    original_stat = df.loc[df['Name'] == player_name, stat]
    if not original_stat.empty:
        # Convert the additional_stat to float or int based on the original stat type
        if df[stat].dtype == 'float64':
            additional_stat = float(additional_stat)
        else:
            additional_stat = int(additional_stat)
        
        # Update the statistic
        df.loc[df['Name'] == player_name, stat] = original_stat.iloc[0] + additional_stat
        # Sort and re-rank after updating
        df.sort_values(by=stat, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['Rank'] = range(1, len(df) + 1)
    else:
        print(f"No data found for {player_name} in {stat}")

# Make sure to convert numeric data properly after loading
for df in dataframes.values():
    for col in ['R', 'HR', 'RBI', 'WAR']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if col in ['R', 'HR', 'RBI']:  # Convert to integer if it's a whole number stat
                df[col] = df[col].astype(int)

# Now you can run the leaderboard display functions without encountering the type error
display_leaderboard_changes('Ted Williams', 'RBI')
display_leaderboard_changes('Joe DiMaggio', 'RBI')

#%%

import pandas as pd
from fuzzywuzzy import process
import os

# Function to load and clean data
def load_and_clean_data(filename, columns, rename_map):
    df = pd.read_csv(os.path.join(PATH, filename))
    cleaned_df = df[columns].copy()
    cleaned_df.columns = rename_map
    # Ensure names are consistent
    cleaned_df['Name'] = cleaned_df['Name'].str.replace('[^a-zA-Z ]', '', regex=True)
    return cleaned_df

# Define the path and load the data
PATH = '/Users/shane/Documents/data_science/what_if/'
cleaned_R = load_and_clean_data('R.csv', ['#', 'Name', 'R'], ['Rank', 'Name', 'R'])
cleaned_HR = load_and_clean_data('HR.csv', ['#', 'Name', 'HR'], ['Rank', 'Name', 'HR'])
cleaned_RBI = load_and_clean_data('RBI.csv', ['#', 'Name', 'RBI'], ['Rank', 'Name', 'RBI'])
cleaned_WAR = load_and_clean_data('WAR.csv', ['#', 'Name', 'WAR'], ['Rank', 'Name', 'WAR'])

# Convert all relevant fields to numeric types to prevent type errors
stat_dataframes = [cleaned_R, cleaned_HR, cleaned_RBI, cleaned_WAR]
for df in stat_dataframes:
    for col in ['R', 'HR', 'RBI', 'WAR']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# Load simulated data
df_ted_sim = pd.read_csv(os.path.join(PATH, 'Ted_Williams_Simulated_Stats_1943-1945.csv'))
df_joe_sim = pd.read_csv(os.path.join(PATH, 'Joe_DiMaggio_Simulated_Stats_1943-1945.csv'))

# Fuzzy match names to update
matches = {name: process.extractOne(name, cleaned_WAR['Name'])[0] for name in ['Joe DiMaggio', 'Ted Williams']}

# Dictionary to store pre and post ranks
pre_post_ranks = {}

# Update all statistics and recalculate ranks
for df, stat in zip(stat_dataframes, ['R', 'HR', 'RBI', 'WAR']):
    # Get pre-update ranks
    df.sort_values(by=stat, ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Rank'] = df.index + 1
    pre_post_ranks[stat] = {}
    for player in ['Ted Williams', 'Joe DiMaggio']:
        pre_post_ranks[stat][player] = {'pre': df[df['Name'] == matches[player]]['Rank'].iloc[0]}
    
    # Update stats
    extra_stat_joe = df_joe_sim[stat].sum()
    extra_stat_ted = df_ted_sim[stat].sum()
    df.loc[df['Name'] == matches['Joe DiMaggio'], stat] += extra_stat_joe
    df.loc[df['Name'] == matches['Ted Williams'], stat] += extra_stat_ted
    
    # Get post-update ranks
    df.sort_values(by=stat, ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Rank'] = df.index + 1
    for player in ['Ted Williams', 'Joe DiMaggio']:
        pre_post_ranks[stat][player]['post'] = df[df['Name'] == matches[player]]['Rank'].iloc[0]

# Print the pre and post ranks for each statistic
print()
for stat, ranks in pre_post_ranks.items():
    print(f"--- {stat} ranks changes: ---")
    for player, rank_changes in ranks.items():
        print(f"{player}: from {rank_changes['pre']} to {rank_changes['post']}")
        
#%%

joe_reg_r = joe_stats['R'].sum()
joe_reg_hr = joe_stats['HR'].sum()
joe_reg_rbi = joe_stats['RBI'].sum()
joe_reg_war = joe_stats['WAR'].sum()

ted_reg_r = ted_stats['R'].sum()
ted_reg_hr = ted_stats['HR'].sum()
ted_reg_rbi = ted_stats['RBI'].sum()
ted_reg_war = ted_stats['WAR'].sum()

joe_sim_r = df_joe_sim['R'].sum()
joe_sim_hr = df_joe_sim['HR'].sum()
joe_sim_rbi = df_joe_sim['RBI'].sum()
joe_sim_war = df_joe_sim['WAR'].sum()

ted_sim_r = df_ted_sim['R'].sum()
ted_sim_hr = df_ted_sim['HR'].sum()
ted_sim_rbi = df_ted_sim['RBI'].sum()
ted_sim_war = df_ted_sim['WAR'].sum()

# Joe's Real and Simulated Stats
print()
print(f"Joe's real Runs: {joe_reg_r}")
print(f"Joe's simulated Runs: {joe_sim_r}")
print(f"Joe's combined Runs: {joe_reg_r + joe_sim_r}")
print()

print(f"Joe's real Home Runs: {joe_reg_hr}")
print(f"Joe's simulated Home Runs: {joe_sim_hr}")
print(f"Joe's combined Home Runs: {joe_reg_hr + joe_sim_hr}")
print()

print(f"Joe's real RBIs: {joe_reg_rbi}")
print(f"Joe's simulated RBIs: {joe_sim_rbi}")
print(f"Joe's combined RBIs: {joe_reg_rbi + joe_sim_rbi}")
print()

print(f"Joe's real WAR: {round(joe_reg_war, 2)}")
print(f"Joe's simulated WAR: {round(joe_sim_war, 2)}")
print(f"Joe's combined WAR: {round(joe_reg_war + joe_sim_war, 2)}")
print()

print('-'*25)
print()
# Ted's Real and Simulated Stats
print(f"Ted Williams' real Runs: {ted_reg_r}")
print(f"Ted Williams' simulated Runs: {ted_sim_r}")
print(f"Ted Williams' combined Runs: {ted_reg_r + ted_sim_r}")
print()

print(f"Ted Williams' real Home Runs: {ted_reg_hr}")
print(f"Ted Williams' simulated Home Runs: {ted_sim_hr}")
print(f"Ted Williams' combined Home Runs: {ted_reg_hr + ted_sim_hr}")
print()

print(f"Ted Williams' real RBIs: {ted_reg_rbi}")
print(f"Ted Williams' simulated RBIs: {ted_sim_rbi}")
print(f"Ted Williams' combined RBIs: {ted_reg_rbi + ted_sim_rbi}")
print()

print(f"Ted Williams' real WAR: {round(ted_reg_war, 2)-0.1}")
print(f"Ted Williams' simulated WAR: {round(ted_sim_war, 2)}")
print(f"Ted Williams' combined WAR: {round(ted_reg_war + ted_sim_war, 2)}")
print()
