import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_matrix(df):
    fig = px.scatter_matrix(df, dimensions=['LandAverage', 'LandMax', 'LandMin', 'OceanAverage'])
    fig.show()

def correlation_matrix(df):
    Numerical = ['LandAverage', 'LandMax', 'LandMin', 'OceanAverage']
    Corrmat = df[Numerical].corr()
    plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(Corrmat, annot=True, fmt=".2f", linewidth=.5)
    plt.show()

def plot_clusters(df, features, labels, title_prefix):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='LandAverage', y='LandMax').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='LandAverage', y='LandMax', hue=labels)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='LandMax', y='LandMin').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='LandMax', y='LandMin', hue=labels)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='LandMin', y='OceanAverage').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='LandMin', y='OceanAverage', hue=labels)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='OceanAverage', y='LandAverage').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='OceanAverage', y='LandAverage', hue=labels)

    plt.show()
