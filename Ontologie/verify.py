from rdflib import Graph, Namespace, RDF, URIRef
from rdflib.namespace import OWL

ECOMACHINE = Namespace("http://ecoMachineLearning.org/ontologies/2023#") 
ENVO = Namespace("http://purl.ENVOlibrary.org/ENVO/")

# percorso per l'ontologia generata
generated_ontology_path = 'Europa/Risultati/IntegratedOntology.owl'

# Carica l'ontologia generata
g_generated = Graph()
g_generated.parse(generated_ontology_path)

print(f"Numero di tripli nell'ontologia generata: {len(g_generated)}")

# Visualizza le classi nell'ontologia
for s in g_generated.subjects(RDF.type, OWL.Class):
    print(f"Classe: {s}")

# Visualizza le proprietà nell'ontologia
for s in g_generated.subjects(RDF.type, OWL.ObjectProperty):
    print(f"Proprietà: {s}")

# Visualizza gli individui (nazioni) e le loro proprietà
for s in g_generated.subjects(RDF.type, ECOMACHINE.Country):
    print(f"Individuo: {s}")
    for p, o in g_generated.predicate_objects(subject=s):
        print(f"  Proprietà: {p}, Valore: {o}")

# Mappatura dei cambiamenti ai loro URI
disorder_uris = {
    "Land Average": "http://purl.ENVOlibrary.org/ENVO/DOID_5419",
    "Land Max": "http://purl.ENVOlibrary.org/ENVO/DOID_1596",
    "Land Min": "http://purl.ENVOlibrary.org/ENVO/DOID_2030",
    "Ocean Average": "http://purl.ENVOlibrary.org/ENVO/DOID_3312"
}

# Verifica che le relazioni siano corrette
for country in g_generated.subjects(RDF.type, ECOMACHINE.Country):
    for disorder_label, disorder_uri in disorder_uris.items():
        if (country, ECOMACHINE.hasDisorder, URIRef(disorder_uri)) in g_generated:
            print(f"{country}cambiamentoClimatico {disorder_label}")
        else:
            print(f"ERRORE: {country} cambiamentoClimatico{disorder_label}")
