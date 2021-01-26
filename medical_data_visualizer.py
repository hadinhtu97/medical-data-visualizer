import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('data.csv')

df['overweight'] = np.where(
    df['weight'] / pow(df['height'] / 100, 2) > 25, 1, 0)

df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)


def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=[
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    df_cat = df_cat.value_counts(sort=False)
    df_cat = df_cat.reset_index(
        level=['cardio', 'variable', 'value'], name='count')

    fig = sns.catplot(kind='bar', x='variable', y='count',
                      hue='value', col='cardio', data=df_cat).fig
    fig.savefig('./img/catplot.png')
    return fig


def draw_heat_map():
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(
        0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig = plt.subplots(1, 1, figsize=(11, 9))

    fig = sns.heatmap(corr, annot=True, fmt='.1f', mask=mask,
                      vmax=0.24, vmin=0.08, linewidths=0.5).figure

    fig.savefig('./img/heatmap.png')
    return fig
