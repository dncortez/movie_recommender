# Movie Recommender

Modelo de recomendación de películas mediante Deep Learning en base a atributos de las películas, su sinopsis e interacciones de usuarios.

El presente repositorio contiene 2 versiones de modelos de recomendación en base a embeddings. El principio detrás de este tipo de sistemas de recomendación es entrenar embeddings (representaciones vectoriales) que resuman las características de preferencia de los artículos o los usuarios, de manera que artículos (en este caso películas) que los ven los mismos usuarios (o usuarios que ven las mismas películas) tengan embeddings similares.

Cada versión presenta 2 elementos: Los embeddings de películas y los embeddings de usuarios. En el enfoque utilizado primero se entrenaron los embeddings de películas en base a su patrón de usuarios que las ven. Luego, se entrenaron los embeddings de los usuarios de manera tal que con una red neuronal que toma el embedding del usuario y el embedding de la película se obtenga la probabilidad de que el usuario haya visto la película.

En estos modelos, a diferencia de otros donde simplemente se asocia un embedding a cada película y/o usuario, los embeddings se obtienen de las características de las películas (año, género, origen, trama) y de los usuarios (patrón de películas vistas) de forma tal que es posible utilizar el mismo sistema de recomendación para nuevas películas y nuevos usuarios.

## Métodos de recomendación

Con cada modelo existen 2 métodos de recomendar películas en base a un usuario o las películas vistas por el usuario:

- Método por densidad (Bruto): Consiste en simplemente buscar las películas con mayor similitud a alguna de las películas vistas por el usuario. Esto se hace mediante similitud coseno de los embeddings de las películas con los de cada película del usuario. Se asume una distribución normal sobre cada similitud coseno y se suman las densidades.
$$\Large \sum_{User} e^{-\frac{cos(E_{c},E_{u})^2}{s\sigma^2}}$$

## Métricas

Se utilizaron 2 métricas para 
