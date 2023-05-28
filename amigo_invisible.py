import pandas as pd
import random

datafr = pd.read_csv(r'C:\Users\metol\Documents\amigo_invisible.csv',sep=';')
datafr = datafr.sort_values(by='dificultad',ascending=False)
datafr = datafr.reset_index(drop=True)


# People should not have as invisible friend their partner or siblings
people = datafr['nombre']
ya_tiene = set()
for index in range(len(datafr)):
    posibles = set(people) - set(eval(datafr['Exclusiones'][index])) - ya_tiene
    sorteo = random.sample(posibles,1)[0]
    ya_tiene.add(sorteo)
    datafr.at[index, 'Regala a'] = sorteo

print(datafr[datafr['Regala a']!='Gabriel'])

print("Espera a Sara")

print(datafr[datafr['Regala a']=='Gabriel'])

print("Bye")
