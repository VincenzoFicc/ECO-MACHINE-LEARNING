import os
import pandas as pd
import data_visualization as dv
import optimize_rf as opt

# percorso del file corrente
base_dir = os.path.dirname(os.path.abspath(__file__))

# Lettura dei dati da file CSV
Data1 = pd.read_csv(os.path.join(base_dir, '..', 'DbDefinitivi', 'ClimateChange.csv'))
Data2 = pd.read_csv(os.path.join(base_dir, '..', 'DbDefinitivi', '1- GlobalLandTemperaturesByCity.csv'))
Data3 = pd.read_csv(os.path.join(base_dir, '..', 'DbOriginali', '5- GlobalTemperatures.csv'))

# Creazione dei DataFrame
df1 = pd.DataFrame(Data1)
df2 = pd.DataFrame(Data2)
df3 = pd.DataFrame(Data3)

# Filtrare i dati per l'Italia
data_italy = Data1[Data1['Entity'] == 'Italy']
# Elenco di tutte le metriche presenti nel dataset
all_columns = Data1.columns.tolist()
# Rimuovere le colonne che non sono metriche (Entity, Code, Year)
feature_columns = [col for col in all_columns if col not in ['Entity', 'Code', 'Year'] + dalys_columns]
# Aggiungere l'anno come feature
feature_columns.append('Year')

def main_menu():
    while True:
        print("\nMenu Principale:")
        print("1. Analisi Descrittiva del Dataset")
        print("2. Elaborazione dei Dati")
        print("0. Esci")

        main_choice = input("Inserisci la tua scelta: ")

        if main_choice == "1":
            while True:
                print("\nAnalisi Descrittiva del Dataset:")
                print("1. Grafico per visualizzare la tendenza media dei cambiamenti climatici")
                print("2. Grafico a barre: tempertura massima e minima")
                print("3. Grafico a barre: temperature")
                print("4. Matrice di Correlazione")
                print("5. Nazioni con incremento di cambiamento climatico")
                print("6. Subplot: temperatura maggiore e minore")
                print("7. Grafico a Linee: temperatura maggiore")
                print("8. Grafico a Linee: temperatura minore")
                print("9. Box Plot")
                print("0. Torna al Menu Principale")

                choice = input("Inserisci la tua scelta: ")

                if choice == "1":
                    dv.tendency_climate_change(df1)
                elif choice == "2":
                    dv.tendency_LandAverage_LandMax(df1)
                elif choice == "3":
                    dv.plot_comparative_bar_chart_max_min(df1)
                elif choice == "4":
                    dv.correlation_heatmap(df1)
                elif choice == "5":
                    dv.analyze_and_plot_trends(df1)
                elif choice == "6":
                    dv.subplot_min_max(df2)
                elif choice == "7":
                    dv.line_chart_max(df3)
                elif choice == "8":
                    dv.line_chart_min(df3)
                elif choice == "9":
                    dv.box_plots(df1)
                elif choice == "0":
                    break
                else:
                    print("Scelta non valida. Per favore, riprova.")

        elif main_choice == "2":
            while True:
                print("\nElaborazione dei Dati:")
                print("1. Previsione cambaimenti climatici in Italia")
                print("0. Torna al Menu Principale")

                choice = input("Inserisci la tua scelta: ")

                if choice == "1":
                        opt.predict_dalys(feature_columns, dalys_columns, data_italy)
                elif choice == "0":
                    break
                else:
                    print("Scelta non valida. Per favore, riprova.")

        elif main_choice == "0":
            break
        else:
            print("Scelta non valida. Per favore, riprova.")


if __name__ == "__main__":
    main_menu()