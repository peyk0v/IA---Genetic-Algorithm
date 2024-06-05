from deap import base, creator, tools
import numpy
import pandas as pd
import random
from deap import algorithms

# Carga los datos del JSON
data = pd.read_json('test.json')

# Define el individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_materia", random.choice, data['Código'].tolist())
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_materia, n=3)

dias_libres = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
preferencia_carga = "Practica"

# Define la función de aptitud
def evaluar(individual):
    score = 0
    troncal_included = False
    materias_seleccionadas = []

    for code in individual:
        # Busca la materia en los datos cargados del JSON
        materia = data[data["Código"] == code]
        if not materia.empty:
            # Verifica si el día de la materia está en los días libres del estudiante
            if any(dia in dias_libres for dia in materia["Día"].values[0]):
                score += 1

            # Aumenta el puntaje según la preferencia de carga del estudiante
            if preferencia_carga == "Teoria":
                if materia["Teoria"].values[0] == "alta":
                    score += 1
                if materia["Practica"].values[0] == "alta":
                    score -= 1
            elif preferencia_carga == "Practica":
                if materia["Practica"].values[0] == "alta":
                    score += 1
                if materia["Teoria"].values[0] == "alta":
                    score -= 1

            # Aumenta el puntaje según el puntaje del profesor
            score += float(materia["Puntaje"].values[0].replace(",", "."))

            # Verifica si la materia es troncal
            if materia["Tipo"].values[0] == "Troncal":
                troncal_included = True

            # Añade la materia a la lista de materias seleccionadas
            materias_seleccionadas.append(materia.to_dict('records')[0])

    # Penaliza si no se incluye una materia troncal
    if not troncal_included:
        score -= 100

    return score, materias_seleccionadas

# Crea un individuo
ind = toolbox.individual()

# Evalúa el individuo
print(evaluar(ind))