# Projecto Machine Learning Operation de Henry


![imagen de muchas peliculas](/src/img/imagen_peliculas.jpg)

### About:
El proyecto sera crear un modelo funcional de recomendaciones de peliculas usando machine learning.

Contamos con un dataset de unas 23 columnas con mas de 45000 filas las cuales tendremos que analizar y limpiar para poder desarrollar una aplicacion que valga la pena.


En este primer proyecto integrador de henry se nos plantean varios desafios que tenemos que resolver.


**tales como:**\
- Hacer un proceso de ETL a nuestro dataset de peliculas
- Hacer un deploy con FastAPI
- Crear 7 endpoints en nuestra app para devolver consultas.
- Hacer un analisis exploratorio de los datos y sacar conclusiones en base a ello
- Crear un modelo de machine learning para implementar en la api.


---
---

![imagen ilustrada de machine learning](/src/img/imagen_ml.png)

Puedes seguir el proceso paso a paso en cada uno de los notebooks, tambien les dejare un poco de mi breve experiencia aca abajo con este dataset y el proyecto completo.



---
# Proceso de ETL

El proceso de ETL fue desafiante y confuso a la vez, una vez que comence con el proyecto pense que en un par de horas estaria resuelto pero dios, cada vez habian mas cosas que no habia visto. Soy una persona un tanto impulsiva pero no de mas, por lo que hice lo correcto cuando elimine todo mi etl el segundo dia y comenze de nuevo con ideas mas concretas.

Una vez que entendi que era lo que realmente necesitaba para hacer el proyecto fue mucho mas facil. Me enfoque directamente en realizar los endpoints y a medida que salian errores o el resultado no era el esperado modificaba mi ETL, por lo que no pase un muy buen rato, sumado a mi leve experiencia con FastAPI.

Por suerte ahora me siento mas confiado con enfrentar una gran cantidad de datos, por lo que este proceso si bien fue desafiante y frustrador, fue divertido y satisfactorio, pero todavia queda el resto del trabajo..

# Creacion de api con FastAPI
Si bien no queria comenzar con esto ya que me obligaba a seguir estudiando herramientas nuevas y no aplicar lo que hasta ahora habia aprendido, pero fue necesario para poder afinar mi proceso de ETL, por suerte encontre material suficiente para poder deployar la api sin mayores problemas.

Tambien la pase un poco mal en el tema de los endpoints ya que me costaba debugear los errores de como leia los tipos de datos mi archivo main, pero tras un rato de pruebas con cada uno pude entender como trabajaba con los dataframes leidos, los "Internal Server Error" me volvian loco pero me ayudaban a mejorar el codigo que escribia asi que pude superar el desafio que brindaba.


# Analisis exploratorio de datos (EDA)
Okey aca voy a ser directo. No sabia por donde arrancar, di mil vueltas por los datos, elimine montones de veces el archivo y comenze de nuevo, luego no me lo tome tan a pecho cuando me puse a analizar el sistema de recomendacion que tenia que crear,  por lo tanto me deje lo mas obvio que eran la cantidad de votos y el promedio de votos, filtrando asi las peliculas que tengan un score mas acertado. Analize tambien un poco los generos y pase directo al machine learning.

# Aprendizaje automatico con NLPK
Tuve varias ideas para hacer el proyecto pero ninguna me gustaba, intente con hacer un cluster de las peliculas y sus generos para recomendar con el tipo de genero y por score pero fracase en el intento, luego tuve la mala idea de intentar filtrar las peliculas por score y genero donde nuevamente fracase por el tamaño de los archivos a la hora de subirlo a github, por lo que termine dejandolo de lado.

Termine optando por un modelo de lenguaje natural para leer los resumenes y devolver las peliculas que tengan contenido parecido, ordenadas por mejores scores asi no defraudamos a nuestro posible cliente. Tras repasar un par de clases y un poco de investigacion pude crear el sistema con la matriz de similitud para hacer las equivalencias entra las distintas peliculas, por lo que resulto y me dio un modelo funcional.

# GitHub
Si bien con github ya tenia cierta experiencia subiendo proyectos y demas en plataformas como freecodecamp, fue desafiante el tema de las capacidades, cometi unos errores con las cargas del .gitignore y me estropie un git entero que tuve que volver a empezar, luego de eso no tuve problemas mas que mirar bien el peso de los archivos a los que les hacia pull. Me ayudo mucho el sistema de versiones que tienen permitiendome trabajar sin estropear mi modelo sin vuelta atras.

# Render
El deploy en render me asustaba por los comentarios que se hacian pero creo que el desafio estuvo en implementar un sistema estable dentro del render, tuve que eliminar y modificar varias versiones de mis librerias para hacer el deploy pero se realizo de una manera exitosa por lo que estoy muy contento de haber conocido esta plataforma.


