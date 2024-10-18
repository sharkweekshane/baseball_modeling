import pybaseball
import pandas as pd
from pybaseball import batting_stats

# Retrieve batting stats from 1900 to 2024
data1 = batting_stats(1860, 1950)
data2 = batting_stats(1951, 2024)
print("Done loading the data...")

#%%

data3 = pd.concat([data1, data2], ignore_index=True)
data3_less = data3[['IDfg', 'Season', 'Name', 'Age', 'PA', 'HR', 'SB', 'wRC+']]

filt0 = data3_less[(data3_less['SB'] > 9) & (data3_less['HR'] > 9)]
filt1 = data3_less[(data3_less['SB'] > 19) & (data3_less['HR'] > 19)]
filt2 = data3_less[(data3_less['SB'] > 29) & (data3_less['HR'] > 29)]
filt3 = data3_less[(data3_less['SB'] > 39) & (data3_less['HR'] > 39)]
filt4 = data3_less[(data3_less['SB'] > 49) & (data3_less['HR'] > 49)]


filt1 = data3_less[(data3_less['SB'] > 19) & (data3_less['HR'] > 19)]
filta = data3[data3['HR'] > 20]
filtb = data3[data3['SB'] > 20]

print(f" Total number of players with 10-10 seasons: {filt0.shape[0]}")
print(f"Number of unique players with 10-10 seasons: {filt0['IDfg'].nunique()}")
print(25*'-')
print(f" Total number of players with 20-20 seasons: {filt1.shape[0]}")
print(f"Number of unique players with 20-20 seasons: {filt1['IDfg'].nunique()}")
print(25*'-')
print(f" Total number of players with 30-30 seasons: {filt2.shape[0]}")
print(f"Number of unique players with 30-30 seasons: {filt2['IDfg'].nunique()}")
print(25*'-')
print(f" Total number of players with 40-40 seasons: {filt3.shape[0]}")
print(f"Number of unique players with 40-40 seasons: {filt3['IDfg'].nunique()}")
print(25*'-')
print(f" Total number of players with 50-50 seasons: {filt4.shape[0]}")
print(f"Number of unique players with 50-50 seasons: {filt4['IDfg'].nunique()}")

import plotly.graph_objects as go
import plotly.io as pio


# Define the stages and values for each stage
stages = ['Stage 1: Awareness', 'Stage 2: Interest', 'Stage 3: Decision', 'Stage 4: Action']
values = [filt0.shape[0], filt1.shape[0], filt2.shape[0], filt3.shape[0], filt4.shape[0]]  # Replace with your actual values

# Create a funnel chart
fig = go.Figure(go.Funnel(
    y = stages,
    x = values,
    textinfo = "value+percent initial"  # Shows values and initial percentage at each stage
))

# Show the figure
fig.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Get the value counts of 'Season'
lala = filt1['Season'].value_counts().head(10)

# Create the bar plot
plt.figure(figsize=(6, 4))
lala.plot(kind='barh', color='darkseagreen')  # 'barh' creates a horizontal bar plot

# Ensure x-axis labels are integers
plt.xticks(np.arange(0, max(lala.values) + 1, step=1))  # Step set to 1 for whole integers

# Add labels and title
plt.xlabel('Count')
plt.ylabel('Season')
plt.title('Top 10 seasons of 20-20 players')

# Show the plot with tight layout and remove spines
plt.tight_layout()
sns.despine()
plt.show()

#%% 

#### line graph that shows the number of 10-10 seasons, 20-20, etc

### line graph that shows the unique number of players with such seasons

#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

bins = [1871, 1880] + list(np.arange(1881, filt1['Season'].max() + 10, 10))

plt.figure(figsize=(10, 6))
plt.hist(filt1['Season'], bins=bins, color='darkseagreen', edgecolor='black')
plt.title('Histogram of 20-20 seasons binned by decade', fontsize=16)
# plt.xlabel('Season', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

decades = np.arange(1880, filt1['Season'].max() + 10, 10)
plt.xticks(ticks=decades, labels=[f"{int(dec)}s" for dec in decades], rotation=45, fontsize=16)

plt.grid(False)
sns.despine()
plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns

sb_thresholds = range(20, 36)
mean_rows_2023_2024 = []

for sb_cutoff in sb_thresholds:
    filt_ = data3_less[(data3_less['SB'] >= sb_cutoff) & (data3_less['HR'] > 19)]
    filtered_2023_2024 = filt_[(filt_['Season'] >= 2023) & (filt_['Season'] <= 2024)]
    mean_rows = len(filtered_2023_2024) / 2
    mean_rows_2023_2024.append(mean_rows)

plt.figure(figsize=(8, 5))
plt.plot(sb_thresholds, mean_rows_2023_2024, marker='o', color='darkseagreen')

for i, value in enumerate(mean_rows_2023_2024):
    plt.text(sb_thresholds[i], value + 0.1, f'{value:.1f}', ha='center', fontsize=10)

plt.axhline(y=8.82, color='plum', linestyle='--', label='Mean = 8.82')

plt.xticks(ticks=range(min(sb_thresholds), max(sb_thresholds) + 1))
plt.yticks(ticks=range(0, int(max(mean_rows_2023_2024)) + 2))

plt.title('2023 & 2024 power-speed thresholding', fontsize=16)
plt.xlabel('SB Cutoff', fontsize=14)
plt.ylabel('Average Number of Rows', fontsize=14)

plt.legend()

plt.grid(False)
sns.despine()
plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns

# Initialize lists to store results
sb_thresholds = range(20, 36)  # SB cutoffs from 20 to 29
mean_rows_2023_2024 = []

# Loop over each SB cutoff
for sb_cutoff in sb_thresholds:
    # Apply the filter with the current SB cutoff (inclusive of the threshold)
    filt_ = data3_less[(data3_less['SB'] >= sb_cutoff) & (data3_less['HR'] > 19)]
    
    # Filter data for 2023 and 2024
    filtered_2023_2024 = filt_[(filt_['Season'] >= 2023) & (filt_['Season'] <= 2024)]
    
    # Calculate the average number of rows for the filtered DataFrame
    mean_rows = len(filtered_2023_2024) / 2  # Divide by 2 to get the average across 2 years
    
    # Store the result
    mean_rows_2023_2024.append(mean_rows)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sb_thresholds, mean_rows_2023_2024, marker='o', color='darkseagreen')

# Add text labels above each point
for i, value in enumerate(mean_rows_2023_2024):
    plt.text(sb_thresholds[i], value + 0.1, f'{value:.1f}', ha='center', fontsize=10)

# Ensure x-axis includes all SB cutoffs
plt.xticks(ticks=range(min(sb_thresholds), max(sb_thresholds) + 1))  # Show all x-axis values

# Set the y-axis to show all integers
plt.yticks(ticks=range(0, int(max(mean_rows_2023_2024)) + 2))  # Set y-axis to show all integers

# Title and labels
plt.title('Average Number of Rows in 2023 and 2024 for Various SB Cutoffs', fontsize=16)
plt.xlabel('SB Cutoff', fontsize=14)
plt.ylabel('Average Number of Rows', fontsize=14)

# Remove grid and adjust aesthetics
plt.grid(False)
sns.despine()
plt.tight_layout()
plt.show()


#%%

x1 = filt1[filt1['Season'] == 2023] # 17 individuals
x2 = filt1[filt1['Season'] == 2024] # 19 individuals


#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming filt1, filta, and filtb are already defined
bins = [1871, 1880] + list(np.arange(1881, max(filt1['Season'].max(), filta['Season'].max(), filtb['Season'].max()) + 10, 10))

# Create a 1x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot for filt1
axes[2].hist(filt1['Season'], bins=bins, color='darkseagreen', edgecolor='black')
axes[2].set_title('20-20 Histogram', fontsize=14)
# axes[0].set_xlabel('Season', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].tick_params(axis='x', rotation=45)

# Plot for filta
axes[1].hist(filta['Season'], bins=bins, color='salmon', edgecolor='black')
axes[1].set_title('20 HR histogram', fontsize=14)
# axes[1].set_xlabel('Season', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

# Plot for filtb
axes[0].hist(filtb['Season'], bins=bins, color='skyblue', edgecolor='black')
axes[0].set_title('20 steal histogram', fontsize=14)
# axes[2].set_xlabel('Season', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

plt.tight_layout()
sns.despine()
plt.show()

#%%

### Who had the most 2020 sesons?

filt1['appearances'] = filt1.groupby('IDfg')['IDfg'].transform('count')

filt1_less = filt1[['IDfg', 'Season', 'Age', 'Name', 'appearances']]




#%%

### What does the age distribution of 2020 seasons look like?

# Counting the occurrences of each age
age_counts = filt1['Age'].value_counts().sort_index()

# Optional: Plotting the counts with thicker, touching, green bars
plt.figure(figsize=(5, 4))
age_counts.plot(kind='bar', color='darkseagreen', width=.9)  # width=1.0 makes bars thicker and touching
plt.title('Age Distribution of 20-20 seasons')
plt.xlabel('Age')
plt.ylabel('Count')
sns.despine()
plt.show()


#%%

# ['IDfg'] == 1109, 1001157, 945

# from filt1, I want you to list every season for this ['IDfg'] == 1109

# List the ['Season'], 'Name', Age, Hr and SB

from great_tables import GT, md

filtered_df1 = filt1[filt1['IDfg'] == 1109][['Season', 'Name', 'Age', 'HR', 'SB']]

filtered_df1['SB'] = filtered_df1['SB'].astype("Int64")

filtered_df1 = filtered_df1.sort_values(by='Season')

gt_table1 = (
    GT(filtered_df1)
    .tab_header(
        title="Barry Bonds",
        subtitle="All his 20-20 seasons"
    )
    # .tab_stub(rowname_col="Season")
    .tab_source_note(
        source_note="Source: pybaseball download Sept 2024"
    )
    .tab_source_note(
        source_note=md("")
    )
    .tab_stubhead(label="Season")
    .fmt_integer(columns="SB")  
)

# Step 5: Save the table
gt_table1.save("Barry_Bonds_with_footer.png")

#%%

from great_tables import GT, md

filtered_df1 = filt1[filt1['IDfg'] == 1001157][['Season', 'Name', 'Age', 'HR', 'SB']]

filtered_df1['SB'] = filtered_df1['SB'].astype("Int64")

filtered_df1 = filtered_df1.sort_values(by='Season')

gt_table1 = (
    GT(filtered_df1)
    .tab_header(
        title="Bobby Bonds",
        subtitle="All his 20-20 seasons"
    )
    # .tab_stub(rowname_col="Season")
    .tab_source_note(
        source_note="Source: pybaseball download Sept 2024"
    )
    .tab_source_note(
        source_note=md("")
    )
    .tab_stubhead(label="Season")
    .fmt_integer(columns="SB")  
)

# Step 5: Save the table
gt_table1.save("Bobby_Bonds_with_footer.png")

#%%


from great_tables import GT, md

filtered_df1 = filt1[filt1['IDfg'] == 945][['Season', 'Name', 'Age', 'HR', 'SB']]

filtered_df1['SB'] = filtered_df1['SB'].astype("Int64")

filtered_df1 = filtered_df1.sort_values(by='Season')

gt_table1 = (
    GT(filtered_df1)
    .tab_header(
        title="Bobby Abreu",
        subtitle="All his 20-20 seasons"
    )
    # .tab_stub(rowname_col="Season")
    .tab_source_note(
        source_note="Source: pybaseball download Sept 2024"
    )
    .tab_source_note(
        source_note=md("")
    )
    .tab_stubhead(label="Season")
    .fmt_integer(columns="SB")  
)

# Step 5: Save the table
gt_table1.save("Bobby_Abreu_with_footer.png")

#%%

import matplotlib.pyplot as plt
import seaborn as sns

# Counting the occurrences of each age
age_counts = filt1['Age'].value_counts().sort_index()

# Defining prime age range (24-30)
prime_age_range = range(25, 29)

# Calculating the sum of counts for prime age and non-prime age
prime_age_counts = age_counts[age_counts.index.isin(prime_age_range)].sum()
non_prime_age_counts = age_counts[~age_counts.index.isin(prime_age_range)].sum()

# Printing the comparison of prime vs non-prime counts
print(f"Prime Age (26-30) Count: {prime_age_counts}")
print(f"Non-Prime Age Count: {non_prime_age_counts}")

# Check if more than half are in the prime range
if prime_age_counts > non_prime_age_counts:
    print("More than half of the 20-20 seasons happen in the prime age range (25-28).")
else:
    print("Less than half of the 20-20 seasons happen in the prime age range (25-28).")

#%%

# from pybaseball import statcast
# import datetime

# # Get today's date in the format YYYY-MM-DD
# today = datetime.datetime.now().strftime('%2024-%09-%29')

# # Get all Statcast data from the earliest point (2015-04-01) until today
# sav = statcast(start_dt='2015-04-01', end_dt=today)

# # Display the first few rows
# print(sav.head())

#%%

# import pandas as pd

# # URL to Baseball Savant sprint speed leaderboard
# url = 'https://baseballsavant.mlb.com/leaderboard/sprint_speed'

# # Read the leaderboard data directly into a pandas DataFrame
# sprint_speed_data = pd.read_html(url)[0]

# # Display the first few rows of sprint speed data
# print(sprint_speed_data.head())

#%%

import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### with a foucs on savant 

filt5 = data3_less[(data3_less['SB'] > 19) & (data3_less['HR'] > 19) & (data3_less['Season'] > 2014)]
print(filt5.shape[0])

filt6 = data3_less[data3_less['Season'] > 2014]
print(filt6.shape[0])