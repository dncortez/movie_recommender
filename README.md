# Movie Recommender

Modelo de recomendación de películas mediante Deep Learning en base a atributos de las películas, su sinopsis e interacciones de usuarios.

El presente repositorio contiene 2 versiones de modelos de recomendación en base a embeddings. El principio detrás de este tipo de sistemas de recomendación es entrenar embeddings (representaciones vectoriales) que resuman las características de preferencia de los artículos o los usuarios, de manera que artículos (en este caso películas) que los ven los mismos usuarios (o usuarios que ven las mismas películas) tengan embeddings similares.

Cada versión presenta 2 elementos: Los embeddings de películas y los embeddings de usuarios. En el enfoque utilizado primero se entrenaron los embeddings de películas en base a su patrón de usuarios que las ven. Luego, se entrenaron los embeddings de los usuarios de manera tal que con una red neuronal que toma el embedding del usuario y el embedding de la película se obtenga la probabilidad de que el usuario haya visto la película.

En estos modelos, a diferencia de otros donde simplemente se asocia un embedding a cada película y/o usuario, los embeddings se obtienen de las características de las películas (año, género, origen, trama) y de los usuarios (patrón de películas vistas) de forma tal que es posible utilizar el mismo sistema de recomendación para nuevas películas y nuevos usuarios.

### Métodos de recomendación

Con cada modelo existen 2 métodos de recomendar películas en base a un usuario o las películas vistas por el usuario:

- Método por densidad (Bruto): Consiste en simplemente buscar las películas con mayor similitud a alguna de las películas vistas por el usuario. Esto se hace mediante similitud coseno de los embeddings de las películas con los de cada película del usuario. Se asume una distribución normal sobre cada similitud coseno y se suman las densidades.
$$\Large \sum_{E_u\in\ User} e^{-\frac{cos(E_{c},E_{u})^2}{s\sigma^2}}$$
Donde $E_c$ es el embedding de la película candidata y $E_u$ es el embedding de una película del usuario. Este método es "Bruto", ya que se podrían tener métricas virtualmente perfectas (obtener como recomendación las mismas películas que el usuario ya ha visto) con $\sigma$ suficientemente bajos, pero arroja malas métricas para predicciones out of sample. Este método sólo requiere de embeddings de películas.
- Máxima verosimilitud: Corresponde en calcular la probabilidad de que un usuario haya visto una película en base a el embedding de la película, el embedding de un usuario y una red neuronal. Se realiza esto para todas las películas y se ordenan de mayor a menor probabilidad. Si la red generaliza lo suficientemente bien, se pueden tener buenas predicciones de lo que al usuario le gustaría ver en base a lo que ya ha visto (out of sample).

### Métricas

Se utilizaron 2 métricas para evaluar el desempeño de los modelos de predicción:
- Precision @ 10: Se toman las 10 primeras recomendaciones y se evalúa que porcentaje de ellas es realmente una película vista por el usuario. El problema de esta métrica es que se puede "Forzar", ya que dado que se conocen las películas que ve un usuario se podrían recomendar esas mismas, por lo que no refleja lo esperado por un recomendador.
- Precision @ 10 Out of sample. Equivalente a Precision @ 10 pero se toma sólo una parte de las películas vistas por el usuario como input y se evalúa que porcentaje de las 10 recomendaciones pertenece al resto de las películas vistas por el usuario. Específicamente, se toma un 20% de las películas vistas por el usuario y se toman las 10 mejores recomendaciones sin contar aquellas películas que ya ha visto el usuario, y se evalúa cuales de estas pertenecen al otro 80%. Esta métrica refleja de mejor manera la capacidad generalizadora del modelo de recomendación.

### Resultados

La versión 2 presentó mejor desempeño en casi todas las métricas, sobretodo en métricas out of sample, de manera tal que sus recomendaciones podrían ser más útiles.

- Usuarios Existentes:
  - fdksflkjd
  - fdkjsfkjds 

# API



##
