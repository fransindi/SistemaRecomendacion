from fastapi import FastAPI
import pandas as pd

#instanciamos la API
app = FastAPI()


#Leemos la tabla:
data = pd.read_csv('data/movies_etl.csv')
df = data.copy()

df.spoken_languages.fillna('N/D')
df.spoken_languages = df.spoken_languages.astype('object')


import joblib
#extraemos el df del modelo de ml
df_ml = pd.read_csv('data/df_ml.csv')
#y con joblib traemos la matriz de similitud
similarity_matrix = joblib.load('pickle/similarity_matrix.pkl')


#Creamos el Root!
@app.get('/')
async def inicio():
    return 'Bienvenidos al sistema de recomendacion de peliculas y mi API, que lo disfruten.'

@app.get('/peliculas_idioma/{idioma}')
async def peliculas_idioma(idioma: str = None):
    """
    Se ingresa el idioma como input y devuelve la cantidad de peliculas
    que fueron estrenadas en ese idioma

    parametros:
    ----------
    idioma: Nombre del idioma tal cual en el dataset.
    """
    #iniciamos un contador vacio
    counter = 0 
    #iteramos el dataframe
    for i in range(0, df.shape[0]):
        row = df.iloc[i, 13]
        if row != 'N/D':
            row = eval(row)
            for id in row:
                if id == idioma:
                    #sumamos uno por cada fila con el idioma
                    counter+=1
    return {f'idioma: {idioma}, cantidad: {counter}'}

@app.get('/peliculas_duracion/{pelicula}')
async def peliculas_duracion(pelicula: str = None):
    """
    Se ingresa el nombre de la pelicula y
    devuelve la duracion y anio de estreno de la misma
    parametros:
    -----------
    pelicula: Nombre de la pelicula tal cual en el dataset.
    """
    #creamos una mascara
    peli = df[df['title'] == pelicula]
    #extraemos los valores.
    duracion = peli.runtime.values[0]
    anio = peli.release_year.values[0]
    return {f'duracion: {duracion}, anio: {anio}, pelicula: {pelicula}'}

@app.get('/franquicia/{franquicia}')
async def franquicia(franquicia: str = None):
    """
    Se ingresa una franquicia y devuelve la cantidad de recaudacion de la misma,
    la suma y el promedio.
    parametros:
    -----------
    franquicia: nombre de la franquicia o la collecion a la que pertenece
    """
    #creamos una mascara y extraemos los valores
    mask = df[df['belongs_to_collection'] == franquicia]
    cantidad = mask.shape[0]
    suma = mask['revenue'].sum()
    promedio = mask['revenue'].mean()
    return {f'cantidad: {cantidad}, suma: {suma}, promedio: {promedio}, franquicia: {franquicia}'}

@app.get('/peliculas_pais/{pais}')
async def peliculas_pais(pais: str = None):
    """
    Se ingresa el nombre de un pais y devuelve
    la cantidad de peliculas que fueron estrenadas 
    o grabadas en ese pais
    parametros:
    -----------
    pais: Nombre del pais tal cual en el dataset.
    """
    counter = 0
    for i in range(0, df.shape[0]):
        row = df.iloc[i, 9]
        if row != 'N/D':
            row = eval(row)
            for id in row:
                if id == pais:
                    counter+=1
    return {f'cantidad: {counter}, pais: {pais}'}


@app.get('/productoras_exitosas/{productora}')
async def productoras_exitosas(produc: str = None):
    """
    Se ingresa el nombre de la productora y devuelve la suma de retorno
    y la cantidad de peliculas en la que estuvieron involucradas.
    parametros:
    ------------
    productora: nombre de la productora tal cual sale en el dataset
    """
    #creamos un array vacio y iteramos buscando los indices
    indices = []
    for i in range(0, df.shape[0]):
        row = df.iloc[i, 8]
        if row != 'N/D':
            row = eval(row)
            for id in row:
                if id == produc:
                    indices.append(i)
    #extraemos los valores usando los indices
    suma = df.iloc[indices].revenue.sum()
    cantidad = df.iloc[indices].shape[0]
    return {f'productora: {produc}, suma: {suma}, cantidad: {cantidad}'}

@app.get('/get_director/{director}')
async def get_director(director: str = None):
    """
    Se ingrea el nombre del director y devuelve una lista con 5 peliculas del mismo
    mas los anios de estreno, el ROI, el retorno y la inversion.
    parametros:
    ----------
    director: nombre del director tal cual el dataset.
    """
    indices = []
    for i in range(0, df.shape[0]):
        row = df.iloc[i, -3]
        if row != 'N/D':
            row = eval(row)
            for id in row:
                if id == director:
                    indices.append(i)
    mask = df.iloc[indices]
    retorno = mask['return'].sum()
    peliculas = list(mask['title'].values)
    anio = list(mask['release_year'].values)
    roi = list(mask['return'].values)
    budget = list(mask['budget'].values)
    revenue = list(mask['revenue'].values)
    return {f'Director: {director}, retorno total: {retorno}, peliculas: {peliculas}, anio: {anio}, retorno: {roi}, budget: {budget}, revenue: {revenue}'}



@app.get('/recomendacion/{titulo}')
async def recomendacion(titulo: str = None):
    """
    La funcion recibe un titulo y segun la similaridad de
    sus resumenes recomendara 5 peliculas
    parametros
    ----------
    titulo: nombre de la pelicula tal cual existe en google. tipo str.
    """
    try:
        #extraemos el index de la pelicula
        movie_index = df_ml[df_ml['title'] == titulo].index[0]
        #extrameos los scores de la pelicula
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        #lo sorteamos para traer los mas frecuentes
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        #Traemos el top 5 de peliculas
        top_scores = similarity_scores[1:5+1]
        #extraemos los indices
        top_movie_indices = [score[0] for score in top_scores]
        #buscamos las peliculas
        top_movies = df_ml['title'].iloc[top_movie_indices]
        #iniciamos un diccionario
        dicc = {}
        #le agregamos los valores
        for i, v in enumerate(top_movies.values):
            dicc[i + 1] = v
        #devolvemos las peliculas
        return {'lista_recomendada': dicc}
    #si la pelicula no existe:
    except: 
        return 'Ingresa el nombre de la pelicula correctamente, ej: Toy Story.'