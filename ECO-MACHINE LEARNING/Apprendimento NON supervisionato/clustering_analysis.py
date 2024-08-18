from sklearn.cluster import AgglomerativeClustering, KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class ClusteringAnalysis:
    def __init__(self):
        pass

    # esegue il clustering agglomerativo
    def agglomerative_clustering(self, df):
        features = ['LandAverage', 'LandMax', 'LandMin', 'OceanAverage']
        cluster_agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
        cluster_agg.fit(df[features])
        labels = cluster_agg.labels_

        # Stampa delle nazioni per ciascun cluster
        df['Cluster'] = labels
        for cluster in range(3):
            print(f"Nazioni nel cluster {cluster}:")
            print(df[df['Cluster'] == cluster]['Entity'].unique())
            print("\n")

        self.plot_clusters(df, features, labels, 'Agglomerative Clustering')
        return df
    
    # esegue il clustering KMeans
    def kmeans_clustering(self, df):
        features = ['LandAverage', 'LandMax', 'LandMin', 'OceanAverage']
        kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)

         # Calcolo dell'inerzia e dei centri di cluster
        inertia = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
            kmeans.fit(df[features])
            inertia.append(kmeans.inertia_)

        # Plot del grafico dell'inerzia
        plt.figure(figsize=(10, 5), dpi=200)
        plt.plot(range(1, 10), inertia, color='purple')
        plt.xticks(range(1, 10))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.axvline(x=3, color='b', label='axvline - full height', linestyle="dashed")
        plt.show()

        # Utilizzo di KneeLocator per trovare il gomito
        KL = KneeLocator(range(1, 10), inertia, curve="convex", direction="decreasing")
        print("Numero ottimale di cluster (elbow method):", KL.elbow)

        # Calcolo del coefficiente di silhouette per diversi numeri di cluster
        silhouette_coefficients = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
            kmeans.fit(df[features])
            score = silhouette_score(df[features], kmeans.labels_)
            silhouette_coefficients.append(score)

        # Plot del coefficiente di silhouette
        plt.figure(figsize=(10, 5), dpi=200)
        plt.plot(range(2, 11), silhouette_coefficients, color='purple')
        plt.xticks(range(2, 11))
        plt.xlabel("Numero di cluster")
        plt.ylabel("Coefficiente di silhouette ")
        plt.show()

        # Esecuzione del clustering KMeans con 3 cluster
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df[features])
        labels_Km = kmeans.labels_

        df['Cluster'] = labels_Km
        for cluster in range(3):
            print(f"Nazioni nel cluster {cluster}:")
            print(df[df['Cluster'] == cluster]['Entity'].unique())
            print("\n")
        self.plot_clusters(df, features, labels_Km, 'KMeans Clustering')
        return df
    
    # Funzione per creare scatter plot con clustering
    def plot_clusters(self, df, features, labels, title_prefix):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        sns.scatterplot(ax=axes[0], data=df, x='LandAverage', y='LandMax').set_title('Senza clustering')
        sns.scatterplot(ax=axes[1], data=df, x='LandAverage', y='LandMax', hue=labels)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        sns.scatterplot(ax=axes[0], data=df, x='LandMax', y='LandMin').set_title('Senza clustering')
        sns.scatterplot(ax=axes[1], data=df, x='LandMax', y='LandMin', hue=labels)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        sns.scatterplot(ax=axes[0], data=df, x='LandMin', y='OceanAverage').set_title('Senza clustering')
        sns.scatterplot(ax=axes[1], data=df, x='LandMin', y='OceanAverage', hue=labels)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        sns.scatterplot(ax=axes[0], data=df, x='OceanAverage', y='LandAverage').set_title('Senza clustering')
        sns.scatterplot(ax=axes[1], data=df, x='OceanAverage', y='LandAverage', hue=labels)

        plt.show()

    # Funzione per calcolare valori maggiori e plot patterns
    def analyze_cluster_patterns(self, df):
        numeric_columns = ['LandAverage', 'LandMax', 'LandMin', 'OceanAverage']
        cluster_means = df.groupby('Cluster')[numeric_columns].mean()

        cluster_means_reset = cluster_means.reset_index().melt(id_vars='Cluster', var_name='Climate change', value_name='Mean Value')

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Climate change', y='Mean Value', hue='Cluster', data=cluster_means_reset, palette='viridis')
        plt.ylabel('Mean Value')
        plt.legend(title='Cluster')
        plt.xticks(rotation=45)
        plt.show()
        return cluster_means
