import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import MultiPolygon

def tendency_climate_change(df1):
    # Seleziona le colonne rilevanti per i cambiamenti climatici e l'anno
    climateChange = ['LandAverage', 'LandMax', 'LandMin', 'OceanAverage']
    years = df1['Year'].unique()

    # Calcola la prevalenza media di ogni cambiamento per ciascun anno 
    trend_data = df1.groupby('Year')[climateChange].mean()

    #grafico tendenze
    plt.figure(figsize=(14, 8))

    for climateChange in climateChange:
       plt.plot(trend_data.index, trend_data[climateChange], label=climateChange)

    plt.xlabel('Anno')
    plt.ylabel('Cambiamento climatico')
    plt.title('Trend del cambaimento climatico nel mondo dal 1990 al 2019')
    plt.legend()
    plt.grid(True)
    plt.show()

#tendenza non media di temperatura media e massima (che mostrano correlazione negativa)
def tendency_LandAverage_LandMax(df1):
    #anni specifici
    selected_years = [1990, 1995, 2000, 2005, 2010, 2019]
    df_filtered = df1[df1['Year'].isin(selected_years)]

    # Calcola la prevalenza media di temperatura media e massima per gli anni selezionati
    mean_values = df_filtered.groupby('Year')[['LandAverage', 'LandMax']].mean().reset_index()
    
    #grafico a barre
    bar_width = 0.35
    index = range(len(selected_years))

    plt.figure(figsize=(12, 6))

    bar1 = plt.bar(index, mean_values['LandAverage'], bar_width, label='LandAverage')
    bar2 = plt.bar([i + bar_width for i in index], mean_values['LandMax'], bar_width, label='LandMax')

    plt.xlabel('Anno')
    plt.ylabel('Cambiamento climatico')
    plt.xticks([i + bar_width / 2 for i in index], selected_years)
    plt.legend()
    plt.grid(True)
    plt.show()

# Funzione per visualizzare l'istogramma comparativo dell'andamento medio della temperatura massima e minima negli anni specifici
def plot_comparative_bar_chart_max_min(df1):
    # Seleziona gli anni specifici
    selected_years = [1990, 1995, 2000, 2005, 2010, 2019]
    df_filtered = df1[df1['Year'].isin(selected_years)]

    # Calcola la prevalenza media della temperatura massima e minima per gli anni selezionati
    mean_values = df_filtered.groupby('Year')[['LandMax', 'LandMin']].mean().reset_index()
    
    #Grafico a barre comparativo
    bar_width = 0.35
    index = range(len(selected_years))

    plt.figure(figsize=(12, 6))

    bar1 = plt.bar(index, mean_values['LandMax'], bar_width, label='LandMax')
    bar2 = plt.bar([i + bar_width for i in index], mean_values['LandMin'], bar_width, label='LandMin')

    plt.xlabel('Year')
    plt.ylabel('Mean Prevalence')
    plt.title('Confronto dell\'andamento medio del cambiamento climatico')
    plt.xticks([i + bar_width / 2 for i in index], selected_years)
    plt.legend()
    plt.grid(True)
    plt.show()

#Matrice di correlazione
def correlation_heatmap(df1):
    df1_variables = df1[["LandAverage", "LandMax", "LandMin", "OceanAverage"]]
    Corrmat = df1_variables.corr()
    plt.figure(figsize=(10, 8), dpi=200)
    sns.heatmap(Corrmat, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matrice di Correlazione per il cambiamento climatico')
    plt.show()


# Funzione per calcolare la pendenza della prevalenza nel tempo per una nazione e un cambiamento 
def calculate_trend(df1, country, climateChange):
    country_data = df1[df1['Entity'] == country]
    slope, intercept = np.polyfit(country_data['Year'], country_data[climateChange], 1)
    return slope

# Funzione per analizzare e visualizzare i trend dei cambiamenti
def analyze_and_plot_trends(df1):
    climateChanges = ['LandAverage', 'LandMax', 'LandMin', 'OceanAverage']
    trends_list = []

    for country in df1['Entity'].unique():
        for climateChange in climateChanges:
            trend = calculate_trend(df1, country, climateChange)
            trends_list.append({'Country': country, 'climateChange': climateChange, 'Trend': trend})

    trend_data = pd.DataFrame(trends_list)
    increasing_trends = trend_data[trend_data['Trend'] > 0]

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    trend_geo = world.set_index('name').join(increasing_trends.set_index('Country'))
    trend_geo['geometry'] = trend_geo['geometry'].apply(lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x]))

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'choropleth'}]]
    )

    colors = {
        'LandAverage': 'Blues',
        'LandMax': 'Oranges',
        'LandMin': 'Greens',
        'OceanAverage': 'Reds'
    }

    for climateChange, color in colors.items():
        climateChange_data = trend_geo[trend_geo['climateChange'] == climateChange]
        if not climateChange_data.empty:
            fig.add_trace(
                go.Choropleth(
                    locations=climateChange_data.index,
                    z=climateChange_data['Trend'],
                    locationmode='country names',
                    colorscale=color,
                    showscale=False,
                    name=climateChange
                )
            )

    # Definizione della legenda con colori esadecimali
    color_hex = {
        'Blues': '#1f77b4',
        'Oranges': '#ff7f0e',
        'Greens': '#2ca02c',
        'Reds': '#d62728'
    }

    legend_annotations = [
        dict(
            x=0.98,
            y=0.95 - (i * 0.05),
            xref='paper',
            yref='paper',
            showarrow=False,
            text=f"<b>{climateChange}</b>",
            bgcolor=color_hex[color],
            opacity=0.8,
            bordercolor='black',
            borderwidth=1
        ) for i, (climateChange, color) in enumerate(colors.items())
    ]

    fig.update_layout(
        title_text='Nazioni con incidenza di cambiamento climatico in crescita',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        margin=dict(r=20, t=40, l=20, b=20),
        annotations=legend_annotations
    )

    fig.show()
#Subplot
def subplot_min_max(df2):
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001)
    x1 = ["Andean Latin America", "West Sub-Saharan Africa", "Tropical Latin America", "Central Asia", "Central Europe",
          "Central Sub-Saharan Africa", "Southern Latin America", "North Africa/Middle East",
          "Southern Sub-Saharan Africa",
          "Southeast Asia", "Oceania", "Central Latin America", "Eastern Europe", "South Asia",
          "East Sub-Saharan Africa",
          "Western Europe", "World", "East Asia", "Caribbean", "Asia Pacific", "Australasia", "North America"]

    fig.append_trace(go.Bar(x=df2["Land min"], y=x1, marker=dict(color='rgba(50, 171, 96, 0.6)',
                                                                         line=dict(color='rgba(20, 10, 56, 1.0)',
                                                                                   width=0)),
                            name='Land Min', orientation='h'), 1, 1)

    fig.append_trace(go.Scatter(x=df2["Land max"], y=x1, mode='lines+markers', line_color='rgb(40, 0, 128)',
                                name='Land max'), 1, 2)

    fig.update_layout(
        title='Max and Min Land',
        yaxis=dict(showgrid=False, showline=False, showticklabels=True, domain=[0, 0.85]),
        yaxis2=dict(showgrid=False, showline=True, showticklabels=False, linecolor='rgba(102, 102, 102, 0.8)',
                    linewidth=5, domain=[0, 0.85]),
        xaxis=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0, 0.45]),
        xaxis2=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0.47, 1], side='top',
                    dtick=10000),
        legend=dict(x=0.029, y=1.038, font_size=10),
        margin=dict(l=100, r=20, t=70, b=70),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        annotations=[dict(xref='x2', yref='y2', y=xd, x=ydn + 10, text='{:,}'.format(ydn) + '%',
                          font=dict(family='Arial', size=10, color='rgb(128, 0, 128)'), showarrow=False)
                     for ydn, yd, xd in zip(df2["Land Max"], df2["climateChange"], x1)] +
                    [dict(xref='x1', yref='y1', y=xd, x=yd + 10, text=str(yd) + '%',
                          font=dict(family='Arial', size=10, color='rgb(50, 171, 96)'), showarrow=False)
                     for ydn, yd, xd in zip(df2["Land Min"], df2["climateChange"], x1)] +
                    [dict(xref='paper', yref='paper', x=-0.2, y=-0.109,
                          font=dict(family='Arial', size=20, color='rgb(150,150,150)'), showarrow=False)]
    )
    fig.show()
