import pandas as pd

# Open CSV file
df = pd.read_csv('data/train.csv')

# Split the male and female data
male_df = df[df['label'] == 'male']
female_df = df[df['label'] == 'female']

# Save to separate CSV files
male_df.to_csv('data/male_data.csv', index=False)
female_df.to_csv('data/female_data.csv', index=False)