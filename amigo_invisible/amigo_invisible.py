import ast
from datetime import datetime
import pandas as pd
import random

global TABLA_BASE

BASE_PATH = "C:/git/little_personal_projects/amigo_invisible/"  # C:/Users/metol/Documents/amigo_invisible/"
YEAR = datetime.now().year

def convert_to_list(s):
    return ast.literal_eval(s)

def initialize():
    tabla = TABLA_BASE
    tabla = tabla.sort_values(by='dificultad',ascending=False)
    tabla = tabla.reset_index(drop=True)

    # People should not have as invisible friend their partner or siblings
    people = tabla['nombre']
    ya_tiene = set()
    return tabla, people, ya_tiene

def exclude_previous_years(tabla, last_results):
    for index in range(len(last_results)):
        try:
            king = last_results['nombre'][index]
            peasant = last_results['Regala a'][index]
            king_index = tabla.index[tabla['nombre'] == king].tolist()[0]
            tabla.loc[king_index, 'Exclusiones'].append(peasant)
            print(f"Este año {king} no regala a {peasant}")
        except:
            print(f"Algo ha pasado entre {king} y {peasant}. Seguramente alguno de los dos no regala este año o no regaló el pasado.")
            continue
    return tabla

converters = {'Exclusiones': convert_to_list}
original = pd.read_csv(BASE_PATH + 'amigo_invisible' + '.csv',sep=';', converters=converters)
last_year = pd.read_csv(BASE_PATH + 'resultado_amigo_invisible_' + str(YEAR - 1) + '.csv',sep=';', converters=converters)
two_years_ago = pd.read_csv(BASE_PATH + 'resultado_amigo_invisible_' + str(YEAR - 2) + '.csv',sep=';', converters=converters)
TABLA_BASE = pd.read_csv(BASE_PATH + 'amigo_invisible_' + str(YEAR) + '.csv',sep=';', converters=converters)

TABLA_BASE = exclude_previous_years(TABLA_BASE, last_year)
TABLA_BASE = exclude_previous_years(TABLA_BASE, two_years_ago)

tabla, people, ya_tiene = initialize()

print("Personas que faltan con respecto al original: ", set(original['nombre']) - set(people))
print("Personas que no estaban en el original: ", set(people) - set(original['nombre']))
tries = 0

while True:
    try:
        for index in range(len(tabla)):
            king = tabla["nombre"].loc[index]
            posibles = set(people) - set(tabla['Exclusiones'][index]) - ya_tiene
            if king in tabla['Regala a'].values:
                match_row = tabla.loc[tabla['Regala a'] == king]
                kingsPeasant = match_row["nombre"].iloc[0]
                posibles = posibles - set(kingsPeasant)
            sorteo = random.sample(posibles, 1)[0]
            ya_tiene.add(sorteo)
            tabla.at[index, 'Regala a'] = sorteo
        break
    except:
        print("Con esta combinación, alguien se quedaba sin regalo.")
        tabla, people, ya_tiene = initialize(last_year)
        tries += 1
        if tries < 20:
            continue
        else:
            print("No doy con la combinación perfecta...")
            exit()

# Comprobar 3 veces si hay continuidad en los regalos de principio a fin
for j in range(3):
    primer_king = random.sample(list(tabla["nombre"].values), 1)[0]
    king = primer_king
    already_kings = [primer_king]
    for i in range(len(tabla)):
        peasant = tabla.loc[tabla["nombre"] == king]["Regala a"].values[0]
        already_kings.append(king)
        if not peasant in already_kings:
            king = peasant
        else:
            print(f"Se pierde la continuidad empezando por {primer_king} en el regalo {i}.")
            print(f"Se quedarían sin participar {len(tabla)-i} primos")
            break


print(tabla[tabla['Regala a']!='Gabriel'])

print("Va a salir el nombre de la persona que regala a Gabriel.")

print(tabla[tabla['Regala a']=='Gabriel']["nombre"].values[0])

print("Guardando el resultado para evitar repetirlo el año que viene.")
tabla.to_csv(BASE_PATH + 'resultado_amigo_invisible_' + str(YEAR) + '.csv',sep=';')

print("Bye")
