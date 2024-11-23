import ast
import pandas as pd
import random

global TABLA_BASE

def convert_to_list(s):
    return ast.literal_eval(s)

def initialize(last_year):
    tabla = TABLA_BASE
    tabla = tabla.sort_values(by='dificultad',ascending=False)
    tabla = tabla.reset_index(drop=True)

    # People should not have as invisible friend their partner or siblings
    people = tabla['nombre']
    ya_tiene = set()
    return tabla, people, ya_tiene

def exclude_last_year(tabla, last_results):
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
original = pd.read_csv(r'C:\Users\metol\Documents\amigo_invisible\amigo_invisible.csv',sep=';', converters=converters)
last_year = pd.read_csv(r'C:\Users\metol\Documents\amigo_invisible\resultado_amigo_invisible_2022.csv',sep=';', converters=converters)
TABLA_BASE = pd.read_csv(r'C:\Users\metol\Documents\amigo_invisible\amigo_invisible_2023.csv',sep=';', converters=converters)
TABLA_BASE = exclude_last_year(TABLA_BASE, last_year)

tabla, people, ya_tiene = initialize(last_year)

print("Personas que faltan con respecto al original: ", set(original['nombre']) - set(people))
print("Personas que no estaban en el original: ", set(people) - set(original['nombre']))
tries = 0

while True:
    try:
        for index in range(len(tabla)):
            posibles = set(people) - set(tabla['Exclusiones'][index]) - ya_tiene
            sorteo = random.sample(posibles,1)[0]
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

print(tabla[tabla['Regala a']!='Gabriel'])

print("Va a salir el nombre de la persona que regala a Gabriel.")

print(tabla[tabla['Regala a']=='Gabriel'])

print("Guardando el resultado para evitar repetirlo el año que viene.")
tabla.to_csv(r'C:\Users\metol\Documents\amigo_invisible\resultado_amigo_invisible_2023.csv',sep=';')

print("Bye")
