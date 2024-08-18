import os
import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD, OWL

# Caricamento del dataset
current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, '..', 'Risultati', 'CambiamentoClimatico-GruppoDiIntervento.csv')

df = pd.read_csv(file_path)

# namespace
ECOMACHINE = Namespace("http://ecoMachineLearning.org/ontologies/2023#") 
ENVO = Namespace("http://purl.ENVOlibrary.org/ENVO/")

# Mappatura dei cambiamenti climatici ai loro URI
climateCh = {
    "Land Average": ENVO.DOID_5419,
    "Land Max": ENVO.DOID_1596,
    "Land Min": ENVO.DOID_2030,
    "Ocean Average": ENVO.DOID_3312
}

# Crea un nuovo grafo per l'ontologia integrata
g = Graph()

# Associa i namespace
g.bind("ecoMachine", ECOMACHINE)
g.bind("ENVO", ENVO)

# Importa l'ontologia esistente
existing_ontology_path = os.path.join(current_dir, 'ClimateChangeOntology.owl')
g.parse(existing_ontology_path)

# Funzione per creare RDF
def create_rdf_triples(row):
    country = URIRef(ECOMACHINE + row['Entity'].replace(" ", "_"))
    year = Literal(row['Year'], datatype=XSD.gYear)
    group = row[
        'Gruppo_di_intervento (0: "sviluppo economico: "Alto aumento dlle temperature medie e minime terrestri", 1: "reddito alto: moderato aumento dlle temperature medie e minime terrestri", 2: "Reddito basso: variazioni moderate delle temperature")']

    g.add((country, RDF.type, ECOMACHINE.Country))
    g.add((country, ECOMACHINE.hasYear, year))

    for climateCh, uri in climateCh_uris.items():
        g.add((country, ECOMACHINE.hasClimateChange, uri))

    if group == 0:
        group_label = ECOMACHINE.EconomicDevelopment
    elif group == 1:
        group_label = ECOMACHINE.HighIncome
    else:
        group_label = ECOMACHINE.LowIncome

    g.add((country, ECOMACHINE.belongsToGroup, group_label))

    # Aggiungi altre colonne come proprietà
    g.add((country, ECOMACHINE.LandAverage, Literal(row['Land Average'])))
    g.add((country, ECOMACHINE.LandMax, Literal(row['Land Max'])))
    g.add((country, ECOMACHINE.LandMin, Literal(row['Land Min'])))
    g.add((country, ECOMACHINE.OceanAverage, Literal(row['Ocean Average'])))
    g.add((country, ECOMACHINE.clusterKMeans, Literal(row['Cluster_KMeans'])))


# Crea RDF per tutte le righe del dataset
df.apply(create_rdf_triples, axis=1)

# Serializza il grafo in un file OWL
output_path = os.path.join(current_dir, '..', 'Risultati', 'IntegratedOntology.owl')
g.serialize(destination=output_path, format='xml')
