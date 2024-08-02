#-------- EXAMEN PRACTICO - DSIS -------
#NOMBRE: Kevin Zurita
#PARALELO: SIR-S7-P2
#FECHA: 01/08/2024

from transformers import pipeline
from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app) # Para permitir solicitudes http

@app.route('/')
def aplicacion():
    return render_template("index.html")


@app.route('/clasificacion', methods=['POST'])
def prediccion():
    promt = request.form['input_usuario'] #El usuario ingresa el texto
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    #Ingresamos las etiquetas con las que el programa procede a clasificar
    candidate_labels = ['Politica', 'Religión','Cine']

    prediccion = classifier(promt, candidate_labels)
    
    prediccion_final = prediccion['labels'][0]
    
    if prediccion_final=='Politica':
        prediccion_Contexto = classifier(promt, candidate_labels)
    elif prediccion_final=='Religión':
        prediccion_Contexto = classifier(promt, candidate_labels)
    elif prediccion_final=='Cine':
        prediccion_Contexto = classifier(promt, candidate_labels)
    else:
        prediccion_final_contexto="No puedo generar una etiqueta, porque solo tengo el entrenamiento en política, religión y cine"

    prediccion_final_contexto=prediccion_Contexto['labels'][0]

    print("")
    print(prediccion_final)
    print(prediccion_Contexto)
    print("")
    print(prediccion_final_contexto)

    return render_template("index.html",
                            clasificacion_final=prediccion_final)

if __name__=="__main__": 
    app.run(debug=True, port=8008)