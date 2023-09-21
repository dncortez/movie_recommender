# MovieNet: Modelo de recomendación de películas

Modelo de recomendación de películas mediante Deep Learning en base a atributos de las películas, su sinopsis e interacciones de usuarios.

### Requerimientos

El uso de los modelos requiere las siguientes dependencias:
- Python 3.9.13 o mayor
- Fastapi 0.103.1 o mayor
- Nltk 3.7 o mayor
- Numpy 1.21.5 o mayor
- Pandas 2.0.2 o mayor
- Scikit-learn 1.2.2 o mayor
- Torch 2.0.1 o mayor
- Tqdm 4.64.1 o mayor
- Uvicorn 0.23.2 o mayor

### Instalación

Para instalar el modelo, usa:
```
git clone https://github.com/dncortez/movienet
```

Esto clonará el repositorio y la mayoría de los pesos de los modelos. Para terminar la instalación se deben descargar 2 modelos adicionales lo cual se puede hacer mediante los siguientes comandos:
```
cd movienet/weights
wget https://drive.google.com/file/d/1llI7jbn31-VpIqFY4xDqam4n4oqX4cMT/view?usp=sharing
wget https://drive.google.com/file/d/16ON1PxOsl9NkxBPSNjklkAxNhNKgux8u/view?usp=sharing
tar -xf MovieNet1.rar
tar -xf MovieNet2.rar
```

Finalmente, para instalar las dependencias se puede ejecutar
```
cd ..
pip install -r requirements.txt
```

# Uso de la api

Los modelos recomendadores se incluyen dentro de una API que facilita su manejo. Para inicializarla ejecuta el siguiente comando dentro de la carpeta del repositorio:
```
uvicorn main:app --reload
```

Luego utiliza tu gestor de api favorito para interactuar con los modelos, o genera requests directamente en el navegador en ```http://127.0.0.1:8000``` y agregando el endpoint y argumentos. La API presenta los siguientes endpoints:

```get_new_user_recommendation```. Endpoint para generar recomendaciones a un usuario nuevo. Argumentos:
- ```movieIds [str]```. Las ID de las películas vistas por el nuevo usuario al que se le darán predicciones, separadas por comas.
- ```model [int]```. La versión del modelo de recomendación a utilizar. Por defecto: 2
- ```n [int]```. El número de recomendaciones a entregar. Por defecto: 8.

```get_current_user_recommendation```. Endpoint para generar recomendaciones a un usuario existente. Argumentos:
- ```userId [str]```. La ID del usuario al que se le generarán recomendaciones.
- ```model [int]```. La versión del modelo de recomendación a utilizar. Por defecto: 2
- ```n [int]```. El número de recomendaciones a entregar. Por defecto: 8.

```get_new_movie_recommendation```. Endpoint para generar una recomendación en base a los datos de una nueva película. Argumentos:
- ```year [int]```. El año de estreno de la película. Por defecto: 1992 (promedio del dataset).
- ```origin [str]```. El origen de la película. Debe ser una de las siguientes opciones ```American```,  ```British```,  ```Canadian```,  ```Australian```,  ```Japanese```,  ```Bollywood```,  ```Hong Kong```,  ```Chinese```,  ```Russian```,   ```South_Korean``` o   ```Other```. Por defecto: American.
- ```genres [str]```. Los géneros que tiene la película, separados por coma. Puede ser uno o más de uno de los siguientes: ```Action```,  ```Adventure```,  ```Animation```,  ```Children```,  ```Comedy```,  ```Crime```,  ```Documentary```,  ```Drama```,  ```Fantasy```,  ```Film-Noir```,  ```Horror```,  ```IMAX```,  ```Musical```,  ```Mystery```,  ```Romance```,  ```Sci-Fi```,  ```Thriller```,  ```War```,  ```Western``` o ```(no genres listed)```. Por defecto: "(no genres listed)".
- ```model [int]```. La versión del modelo de recomendación a utilizar. Por defecto: 2
- ```n [int]```. El número de recomendaciones a entregar. Por defecto: 8.

Tanto ```get_new_user_recommendation``` como ```get_current_user_recommendation``` utilizan máxima verosimilitud para generar recomendaciones, ya que tiene mejores métricas out of sample. ```get_new_movie_recommendation``` por su parte utiliza densidad, ya que es el único método que permite utilizar sólo la información de las películas. 

# Descripción del modelo

El presente repositorio contiene 2 versiones de modelos de recomendación en base a embeddings. El principio detrás de este tipo de sistemas de recomendación es entrenar embeddings (representaciones vectoriales) que resuman las características de preferencia de los artículos o los usuarios, de manera que artículos (en este caso películas) que los ven los mismos usuarios (o usuarios que ven las mismas películas) tengan embeddings similares.

Cada versión presenta 2 elementos: Los embeddings de películas y los embeddings de usuarios. En el enfoque utilizado primero se entrenaron los embeddings de películas en base a su patrón de usuarios que las ven. Luego, se entrenaron los embeddings de los usuarios de manera tal que con una red neuronal que toma el embedding del usuario y el embedding de la película se obtenga la probabilidad de que el usuario haya visto la película.

En estos modelos, a diferencia de otros donde simplemente se asocia un embedding a cada película y/o usuario, los embeddings se obtienen de las características de las películas (año, género, origen, trama) y de los usuarios (patrón de películas vistas) de forma tal que es posible utilizar el mismo sistema de recomendación para nuevas películas y nuevos usuarios.

### Métodos de recomendación

Con cada modelo existen 2 métodos de recomendar películas en base a un usuario o las películas vistas por el usuario:

- Método por densidad (Bruto): Consiste en simplemente buscar las películas con mayor similitud a alguna de las películas vistas por el usuario. Esto se hace mediante similitud coseno de los embeddings de las películas con los de cada película del usuario. Se asume una distribución normal sobre cada similitud coseno y se suman las densidades.
$$\Large \sum_{E_u\in\ User} e^{-\frac{cos(E_{c},E_{u})^2}{s\sigma^2}}$$
Donde $E_c$ es el embedding de la película candidata y $E_u$ es el embedding de una película del usuario. Este método es "Bruto", ya que se podrían tener métricas virtualmente perfectas (obtener como recomendación las mismas películas que el usuario ya ha visto) con $\sigma$ suficientemente bajos, pero arroja malas métricas para predicciones out of sample. Este método sólo requiere de embeddings de películas.
- Máxima verosimilitud (M.V.): Corresponde en calcular la probabilidad de que un usuario haya visto una película en base a el embedding de la película, el embedding de un usuario y una red neuronal. Se realiza esto para todas las películas y se ordenan de mayor a menor probabilidad. Si la red generaliza lo suficientemente bien, se pueden tener buenas predicciones de lo que al usuario le gustaría ver en base a lo que ya ha visto (out of sample).

### Métricas

Se utilizaron 2 métricas para evaluar el desempeño de los modelos de predicción:
- Precision @ 10: Se toman las 10 primeras recomendaciones y se evalúa que porcentaje de ellas es realmente una película vista por el usuario. El problema de esta métrica es que se puede "Forzar", ya que dado que se conocen las películas que ve un usuario se podrían recomendar esas mismas, por lo que no refleja lo esperado por un recomendador.
- Precision @ 10 Out of sample. Equivalente a Precision @ 10 pero se toma sólo una parte de las películas vistas por el usuario como input y se evalúa que porcentaje de las 10 recomendaciones pertenece al resto de las películas vistas por el usuario. Específicamente, se toma un 20% de las películas vistas por el usuario y se toman las 10 mejores recomendaciones sin contar aquellas películas que ya ha visto el usuario, y se evalúa cuales de estas pertenecen al otro 80%. Esta métrica refleja de mejor manera la capacidad generalizadora del modelo de recomendación.

### Resultados

La versión 2 presentó mejor desempeño en casi todas las métricas, sobretodo en métricas out of sample, de manera tal que sus recomendaciones podrían ser más útiles. A continuación se detallan los porcentajes de Precision @ 10 tanto in sample como out of sample.

- Usuarios Existentes:
  - Recomendación por densidad: 45.51% in sample | 7.72% out of sample
  - Recomendación por M.V.: 31.38% in sample | **18.07% out of sample**
- Usuarios Nuevos:
  - Recomendación por densidad: 45% in sample | 10.63% out of sample
  - Recomendación por M.V.: 26.75% in sample | **19.19% out of sample** 
