import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
#df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['overweight'] > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df["cholesterol"] = df["cholesterol"].apply(lambda x:0 if x == 1 else 1)
df["gluc"] = df["gluc"].apply(lambda x:0 if x == 1 else 1)
#df.loc[(df['cholesterol'] ==  1) | (df['gluc'] == 1), ['cholesterol', 'gluc']] = 1
#df.loc[(df['cholesterol'] > 1) | (df['gluc'] > 1), ['cholesterol', 'gluc']] = 0

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).value.agg('count').reset_index(name = 'total')

    # Draw the catplot with 'sns.catplot()'
    fig= sns.catplot(x='variable', y='total', data=df_cat, col='cardio', hue='value', kind='bar').fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df1 = df.loc[(df['ap_lo'] <= df['ap_hi'])]
    df1 = df.loc[(df['ap_lo'] <= df['ap_hi'])]
    df2 = df1.loc[(df['height'] >= df['height'].quantile(0.025))]
    df3 = df2.loc[(df['height'] <= df['height'].quantile(0.975))]
    df4 = df3.loc[(df['weight'] >= df['weight'].quantile(0.025))]
    df_heat = df4.loc[(df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12,12))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, linewidths=1, annot=True, square=True, mask=mask, fmt=".1f", center=0.08, cbar_kws={'shrink':0.5})

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig


