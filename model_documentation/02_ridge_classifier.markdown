# <center><b>Clasificador de Ridge</b></center>

---

## <b>¿Qué es?</b>

`RidgeClassifier` es un modelo lineal de clasificación binaria o multiclase que **adapta la regresión ridge (lineal con regularización L2)** para tareas de clasificación.

A diferencia de modelos como la **regresión logística**, que estiman **probabilidades** y usan funciones como **sigmoide o softmax**, `RidgeClassifier`:

- Aplica una regresión lineal regularizada (`Ridge`).
- Asigna la clase directamente usando el **signo** (binaria) o el **valor máximo** (multiclase) de la predicción continua.

### **Características clave:**

- **Funciona como una regresión lineal** pero se usa para clasificar.
- Utiliza regularización **L2** para evitar sobreajuste (penaliza coeficientes grandes).
- Ideal cuando hay **muchas variables predictoras** o **multicolinealidad**.
- No devuelve probabilidades (`predict_proba` no está disponible).

> En resumen: `RidgeClassifier` **no modela la probabilidad**, sino que ajusta una recta regularizada a los targets y luego decide la clase según el valor resultante.

---

## <b>Formulación matemática</b>

`RidgeClassifier` se basa en adaptar la **regresión lineal regularizada (Ridge)** a un contexto de clasificación. A diferencia de la regresión logística, **no usa una función de probabilidad**, sino que **minimiza el error cuadrático (MSE) con penalización L2**.

### **Clasificación binaria**

Se parte de un conjunto de entrenamiento:

- $X \in \mathbb{R}^{n \times p}$: matriz de características.
- $y \in \{-1, +1\}^n$: etiquetas codificadas como -1 y 1.

El objetivo es encontrar un vector de pesos $w \in \mathbb{R}^p$ que minimice la siguiente función de pérdida:

$$
\mathcal{L}(w) = \|Xw - y\|^2 + \alpha \|w\|^2
$$

Donde:

- $\|Xw - y\|^2$ es el **error cuadrático** (diferencia entre la predicción lineal y la etiqueta).
- $\|w\|^2$ es la **norma L2** (penalización Ridge).
- $\alpha > 0$ es el hiperparámetro de regularización (cuánto penaliza pesos grandes).

> Esta función **no es log-loss**, sino una **pérdida cuadrática regularizada**, como en la regresión Ridge.

La predicción se realiza con la **función signo**:

$$
\hat{y} = \text{sign}(Xw)
$$

- Si $Xw > 0$ ⇒ predice clase +1  
- Si $Xw < 0$ ⇒ predice clase -1

### **Clasificación multiclase**

Para más de dos clases, se aplica la estrategia **One-vs-Rest (OvR)**:

1. Se entrena un clasificador binario para cada clase $k$:
   $$ 
   \mathcal{L}(w_k) = \|Xw_k - y_k\|^2 + \alpha \|w_k\|^2 
   $$
   Donde $y_k$ es un vector binario:  
   - $y_k^{(i)} = 1$ si la observación $i$ pertenece a la clase $k$  
   - $0$ en caso contrario.

2. Para predecir una nueva observación:

$$
\hat{y} = \arg\max_k (Xw_k)
$$

Es decir, se calcula la salida lineal $Xw_k$ para cada clase y se escoge aquella con el valor más alto.

### **Comparación con regresión logística**

| Aspecto                     | RidgeClassifier            | Regresión Logística             |
|----------------------------|----------------------------|----------------------------------|
| Función de pérdida         | MSE con regularización L2  | Log-loss con regularización     |
| Tipo de predicción         | Lineal + argmax/signo      | Probabilidades (sigmoide/softmax) |
| Interpretabilidad          | Alta (coeficientes lineales) | Alta (odds ratio)               |
| Soporte para `predict_proba` | No disponible           | Sí disponible                 |
| Velocidad de entrenamiento | Muy rápida               | Rápida pero más costosa       |

### **¿Por qué funciona como clasificador?**

Aunque minimiza un error cuadrático (típico de regresión), el `RidgeClassifier` **aprende una frontera lineal de decisión**, gracias a:

- La penalización L2 que estabiliza los coeficientes.
- La función signo o `argmax`, que convierte la salida continua en una clase.

Esto lo hace efectivo cuando las clases son **linealmente separables o casi**.

---

## <b>Supuestos y consideraciones</b>

Para que `RidgeClassifier` funcione de manera adecuada y produzca resultados estables, es importante tener en cuenta las siguientes condiciones:

- **Separabilidad lineal aproximada entre clases**  
  > *El modelo aprende una frontera lineal, por lo que funciona mejor cuando las clases pueden separarse (aproximadamente) con una combinación lineal de las variables.*

- **No requiere estimar probabilidades**  
  > *Este modelo no está diseñado para devolver probabilidades calibradas. No usa sigmoide ni softmax, por lo que si necesitas probabilidades confiables, considera regresión logística.*

- **Codificación adecuada de etiquetas**  
  > *Las clases deben estar codificadas numéricamente (como enteros). Internamente, `RidgeClassifier` binariza las clases en un esquema One-vs-Rest para multiclase.*

- **Escalamiento de variables**  
  > *Como la regularización L2 penaliza el tamaño absoluto de los coeficientes, es fundamental que las variables estén en la **misma escala**. De lo contrario, variables con mayor rango dominarán el ajuste.*

- **Independencia de las observaciones**  
  > *Cada fila del dataset debe ser independiente. El modelo no maneja relaciones entre observaciones (como datos secuenciales o agrupados), a menos que se ajuste previamente.*

- **Presencia de outliers afecta fuertemente**  
  > *Dado que se basa en error cuadrático, es sensible a valores atípicos extremos, que pueden distorsionar la frontera de decisión.*

- **Multicolinealidad entre predictores puede ser controlada**  
  > *Una de las fortalezas del modelo: la regularización L2 **reduce la varianza** de los coeficientes cuando hay predictores altamente correlacionados.*

- **Tamaño de muestra razonable para generalizar**  
  > *Aunque es eficiente computacionalmente, es importante tener suficientes muestras por clase, especialmente en clasificación multiclase.*

*A diferencia de la regresión logística, `RidgeClassifier` **no modela probabilidades ni requiere supuestos sobre distribución de errores o residuos**.*

---

## <b>Interpretación del modelo</b>

Una vez entrenado el `RidgeClassifier`, es esencial comprender cómo interpretar sus componentes para tomar decisiones o extraer conocimiento del modelo.

### **Coeficientes $w_j$**

Cada coeficiente $w_j$ representa la **influencia lineal de la variable $x_j$ sobre la predicción**.

$$
\hat{y} = \text{sign}(Xw + b)
$$

- Si $w_j > 0$: un aumento en $x_j$ **aumenta el valor de la función lineal**, acercando la predicción a la clase positiva.
- Si $w_j < 0$: un aumento en $x_j$ **disminuye la función lineal**, acercando la predicción a la clase negativa.
- Si $w_j = 0$: $x_j$ **no tiene influencia** en la predicción (o fue regularizado fuertemente).

> Aunque el modelo no produce probabilidades, los coeficientes indican la **dirección e importancia relativa** de cada variable.

### **Intercepto $b$**

El término independiente (intercepto):

$$
\hat{y} = \text{sign}(Xw + b)
$$

- Desplaza la **frontera de decisión** a lo largo del espacio de características.
- Si $b = 0$, la frontera pasa por el origen.

> Es importante si los datos **no están centrados** (es decir, no tienen media cero).

### **Valores continuos: scores y magnitudes**

Aunque el modelo no devuelve probabilidades, los valores de la predicción continua $Xw$ pueden ser usados como **scores de decisión**:

- **Mayor valor absoluto de $Xw$ ⇒ mayor confianza en la predicción**.
- Ejemplo:
  - $Xw = 2.7$ → predicción positiva con alta confianza.
  - $Xw = -0.1$ → predicción negativa con baja confianza.

> Estos scores pueden usarse para ordenar observaciones, detectar ambigüedad en las clases o construir métricas personalizadas.

### **Multiclase: varios vectores de pesos**

En problemas con más de dos clases ($K$ clases), el modelo entrena un vector de pesos $w_k$ por clase (estrategia One-vs-Rest):

- Para una nueva observación $x$, se calculan las predicciones lineales para cada clase:
  $$
  s_k = x^\top w_k + b_k
  $$
- La clase predicha es:
  $$
  \hat{y} = \arg\max_k s_k
  $$

> Cada clase tiene su **propia frontera lineal**, y gana la que tenga el mayor score.

### **Magnitud de los coeficientes**

La **magnitud de los coeficientes** puede usarse como una medida de **importancia relativa** de las variables:

- Variables con coeficientes más grandes (en valor absoluto) **tienen mayor impacto** en la decisión.
- Sin embargo, debido a la regularización L2, los coeficientes pueden ser **pequeños pero significativos**, sobre todo si hay multicolinealidad.

### **No hay odds ni probabilidades**

A diferencia de la regresión logística:

- **No se calculan odds ni log-odds**.
- **No se puede interpretar $w_j$ como un efecto multiplicativo**.
- **No hay `predict_proba()`**.

> Si necesitas probabilidades bien calibradas, considera usar `LogisticRegression`.

### **En resumen:**

| Elemento               | Qué representa                                             |
|------------------------|------------------------------------------------------------|
| $w_j$                  | Influencia lineal de $x_j$ sobre la predicción             |
| $b$                    | Desplazamiento de la frontera de decisión                  |
| $Xw$                   | Score continuo (usado para clasificar)                     |
| $\arg\max_k (Xw_k)$    | Mecanismo de decisión en multiclase                        |
| $\|w\|$ grande          | Variable importante (afecta mucho la predicción)           |
| Probabilidades       | No están disponibles                                       |

> Aunque `RidgeClassifier` no produce probabilidades, sigue siendo **altamente interpretable** gracias a su estructura lineal regularizada. Es útil en tareas donde interesa saber **qué variables influyen más**, incluso si la clasificación es el objetivo principal.

---

## <b>Implementación en `scikit-learn`</b>

La clase `RidgeClassifier` se encuentra en el módulo `sklearn.linear_model`. Aquí mostramos cómo implementarla y ajustar los hiperparámetros clave:

```python
from sklearn.linear_model import RidgeClassifier

model = RidgeClassifier(
    alpha=1.0,
    fit_intercept=True,
    normalize='deprecated',
    copy_X=True,
    max_iter=None,
    tol=1e-3,
    class_weight=None,
    solver='auto',
    positive=False,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
# Predicción de scores (valores continuos antes del signo)
scores = model.decision_function(X_test)
```

---

## <b>Parámetros Cruciales</b>

Los siguientes hiperparámetros de `RidgeClassifier` **afectan directamente el rendimiento del modelo**. Es clave comprenderlos y ajustarlos correctamente.

### **alpha**

- Controla la **fuerza de la regularización L2** aplicada sobre los coeficientes del modelo.
  
- El modelo minimiza:
  $$
  \mathcal{L} = \|Xw - y\|^2 + \alpha \|w\|^2
  $$

- **`alpha` bajo** ($\to 0$): menor penalización, más riesgo de overfitting (modelo más flexible).  
- **`alpha` alto**: mayor penalización, coeficientes más pequeños, menor varianza (modelo más simple).

**Debe ajustarse por validación cruzada** (`GridSearchCV` o `RidgeCV`).

### **solver**

- Define el **algoritmo numérico** utilizado para encontrar los coeficientes.

| Solver        | Tipo        | Ideal para...                             |
|---------------|-------------|-------------------------------------------|
| `'auto'`      | Automático  | Elige la mejor opción automáticamente     |
| `'svd'`       | Exacto      | Datasets pequeños, sin multicolinealidad  |
| `'cholesky'`  | Directo     | Datos densos y bien condicionados         |
| `'lsqr'`      | Iterativo   | Grandes volúmenes de datos                |
| `'sparse_cg'` | Iterativo   | Datos dispersos (sparse)                  |
| `'sag'`, `'saga'` | Estocásticos | Datasets muy grandes                    |

Si usas solvers iterativos, `tol` y `max_iter` deben ser ajustados cuidadosamente.

### **class_weight**

- Ajusta el **peso relativo de cada clase** en la función de pérdida.
- Muy útil cuando hay **clases desbalanceadas**.

Opciones:

- `'balanced'`: calcula pesos automáticamente según frecuencia.
- `dict`: pesos personalizados, e.g., `{0: 1, 1: 3}`.
- `None`: todas las clases pesan igual.

Mejora la sensibilidad y recall en clases minoritarias.

### **tol**

- Es la **tolerancia para el criterio de parada** en algoritmos iterativos (`sag`, `saga`, etc.).
- Valores pequeños (`1e-4`, `1e-5`) ⇒ más precisión pero mayor tiempo.  
- Valores grandes (`1e-2`) ⇒ menos precisión pero más rapidez.

Ajusta si el modelo no converge o converge muy rápido sin mejorar.

### **max_iter**

- Límite de iteraciones del solver **cuando se usan métodos iterativos**.
- No se usa si el solver es exacto (`svd`, `cholesky`), pero es esencial con `saga`, `sag`, etc.

Si obtienes un warning de no convergencia, aumenta este valor (`500`, `1000`, etc.).

### **positive**

- Restringe los **coeficientes a ser solo positivos** ($w_j \geq 0$).
- Útil cuando:
  - Se exige una relación positiva entre variables y salida (por ejemplo, score de riesgo).
  - Se desea interpretabilidad o cumplimiento normativo.

Solo funciona con `solver='saga'`.

> **Resumen experto:**
>
> - Ajusta `alpha` con validación cruzada.
> - Usa `class_weight='balanced'` en problemas desbalanceados.
> - Cuida `solver`, `tol` y `max_iter` si tus datos son grandes.
> - Considera `positive=True` si tu dominio exige interpretabilidad.

---

## <b>Validaciones Numéricas Internas</b>

Cuando ejecutas `model.fit(X, y)` con `RidgeClassifier`, internamente se lleva a cabo una **optimización numérica** de una función de pérdida regularizada.

### **Función objetivo que se optimiza**

El modelo minimiza la **función de pérdida cuadrática penalizada (Ridge)**:

$$
\mathcal{L}(w) = \|Xw - y\|^2 + \alpha \|w\|^2
$$

- $X \in \mathbb{R}^{n \times p}$: matriz de predictores.
- $y \in \mathbb{R}^n$: vector de etiquetas codificadas como $-1$, $1$ (o one-hot en multiclase).
- $w \in \mathbb{R}^p$: vector de coeficientes del modelo.
- $\alpha$: hiperparámetro de regularización L2.

### **¿Qué sucede en `.fit()`?**

Cuando llamas a `.fit()`, el modelo realiza:

1. **Codificación de las clases**
   - Para binaria: codifica las etiquetas como $-1$ y $1$ (no como 0 y 1).
   - Para multiclase: aplica estrategia **One-vs-Rest (OvR)** → una regresión Ridge por clase.

2. **Resolución del sistema de ecuaciones lineales**
   - Si el solver es **exacto** (`'svd'`, `'cholesky'`), se calcula directamente:
     $$
     w = (X^T X + \alpha I)^{-1} X^T y
     $$
     donde $I$ es la matriz identidad.

   - Si el solver es **iterativo** (`'sag'`, `'lsqr'`, `'saga'`), se utilizan técnicas numéricas para minimizar la función objetivo gradualmente:
     - Se calcula el **gradiente** de la función:
       $$
       \nabla_w \mathcal{L} = 2X^T(Xw - y) + 2\alpha w
       $$
     - Se actualiza $w$ en cada paso hasta cumplir los criterios de convergencia (`tol`, `max_iter`).

3. **Asignación de clases**
   - Una vez obtenidos los coeficientes $w$, se calcula:
     $$
     \hat{y} = \text{sign}(Xw) \quad \text{(binaria)}
     $$
     $$
     \hat{y} = \arg\max_k (Xw_k) \quad \text{(multiclase)}
     $$
   - No se usa ninguna función sigmoide ni softmax.

### **Importante**

- `RidgeClassifier` **no calcula probabilidades**, solo scores lineales.
- El resultado final depende de:
  - La calidad de los predictores $X$ (mejor si están escalados).
  - El valor de `alpha`.
  - La elección del `solver` y su convergencia.

> **En resumen:**  
> `RidgeClassifier` encuentra una solución cerrada (o iterativa) al minimizar una pérdida cuadrática penalizada.  
> Esto lo hace estable incluso cuando hay **colinealidad** o muchas variables, y por eso es popular en contextos con muchos features.

---

## <b>Casos de Uso</b>

Aunque `RidgeClassifier` no es tan popular como modelos más complejos como `LogisticRegression`, `RandomForest` o `XGBoost`, **sigue siendo muy valioso en ciertos contextos específicos**.

Aquí te explico **cuándo este modelo tiene ventajas reales**:

### **Alta dimensionalidad** (`n < p`)

- Situaciones donde tienes **más variables que observaciones** (text mining, genómica, sensores, etc.).
- Ridge regulariza fuertemente y evita overfitting en estos casos.
- Ejemplo:
  - Clasificación de texto con TF-IDF (miles de palabras como features, pocos documentos).

### **Multicolinealidad severa**

- Las regresiones ordinarias se vuelven inestables cuando los predictores están fuertemente correlacionados.
- Ridge estabiliza los coeficientes al penalizar su magnitud.
- Ejemplo:
  - Variables económicas o sociodemográficas con alta correlación.

### **Se requiere alta velocidad de entrenamiento**

- Comparado con modelos no lineales, `RidgeClassifier` es muy rápido, incluso con muchas observaciones.
- Útil para sistemas en producción que requieren tiempos de respuesta bajos.

### **Cuando la interpretabilidad es importante**

- Aunque no genera probabilidades, entrega coeficientes lineales que indican el efecto de cada variable.
- Puedes saber qué variables impulsan la decisión del modelo.

### **Clasificación multiclase con estrategia OvR**

- Soporta múltiples clases usando una estrategia **One-vs-Rest**.
- Más estable que `LogisticRegression` en datasets pequeños o ruidosos con muchas clases.

### **Casos en los que *NO* es recomendado**:

- Si necesitas **probabilidades calibradas**: este modelo no estima probabilidades.
- Si los datos no son **linealmente separables** (ni aproximadamente): podrías preferir modelos no lineales como `SVM`, `XGBoost`, etc.
- Si hay muchos **outliers**: al minimizar el error cuadrático, puede verse afectado por valores extremos.

> **Resumen experto:**
>
> Usa `RidgeClassifier` cuando:
> - Tienes muchos predictores (incluso más que observaciones).
> - Hay correlaciones fuertes entre variables.
> - Necesitas una solución rápida, estable y lineal.
>
> Evítalo si:
> - Requieres probabilidades.
> - Los datos son muy no lineales o contienen outliers fuertes.

---

## <b>Profundización Matemática</b>

### **Función de pérdida con regularización L2**

El clasificador Ridge minimiza una **función de pérdida cuadrática penalizada**:

$$
\mathcal{L}(w) = \|Xw - y\|^2 + \alpha \|w\|^2
$$

- $X$: matriz de predictores.
- $y$: vector de etiquetas transformadas a $\{-1, +1\}$ o codificación OvR.
- $w$: coeficientes del modelo.
- $\alpha$: hiperparámetro de regularización L2.

> Esta función busca ajustar bien los datos (primer término) sin que los coeficientes crezcan demasiado (segundo término).

### **Solución cerrada**

Cuando el número de observaciones es mayor al de variables y $X^T X$ es invertible, el modelo tiene una **solución analítica exacta**:

$$
w = (X^\top X + \alpha I)^{-1} X^\top y
$$

- $I$: matriz identidad.
- Este tipo de solución es **estable incluso si hay colinealidad** en los datos.

### **Gradiente de la pérdida**

El gradiente con respecto a los coeficientes $w$ es:

$$
\nabla_w \mathcal{L} = 2X^\top(Xw - y) + 2\alpha w
$$

Esto es útil si se usa un **método iterativo de optimización** como `sag` o `lsqr`, que requieren solo el cálculo del gradiente para actualizar los pesos.

### **Predicción**

Una vez entrenado, el modelo genera predicciones con:

#### **Clasificación binaria**
$$
\hat{y} = \text{sign}(Xw)
$$

#### **Clasificación multiclase**
$$
\hat{y} = \arg\max_k (X w_k)
$$

Donde $w_k$ es el vector de coeficientes correspondiente a la clase $k$ (estrategia One-vs-Rest).

### **Regularización y su impacto**

- Penaliza grandes valores de $w$:
  $$
  \alpha \|w\|^2 = \alpha \sum_j w_j^2
  $$

- Esto **reduce la varianza del modelo** y controla el overfitting, especialmente útil si hay **muchas variables** o **multicolinealidad**.

- Valores altos de $\alpha$ → coeficientes más pequeños, **modelo más conservador**.

> **Resumen para expertos:**
>
> `RidgeClassifier` usa una pérdida cuadrática penalizada con L2.  
> Tiene solución cerrada (cuando se puede) o se optimiza con métodos numéricos.  
> No predice probabilidades, pero ofrece una clasificación estable, rápida y útil en escenarios de alta dimensionalidad.

---

## <b>Recursos para profundizar</b>

### **Libros**
- *The Elements of Statistical Learning* – Hastie, Tibshirani, Friedman  
  > Explica en detalle la regresión ridge y su rol en clasificación multiclase.

- *An Introduction to Statistical Learning* – James et al.  
  > Capítulos sobre modelos lineales y regularización accesibles para quienes inician.

- *Applied Predictive Modeling* – Kuhn & Johnson  
  > Excelente enfoque aplicado con uso práctico de Ridge y Lasso.

- *Machine Learning: A Probabilistic Perspective* – Kevin Murphy  
  > Ofrece una visión más formal y bayesiana sobre la regresión regularizada.

### **Cursos**
- Andrew Ng – *Machine Learning* (Coursera)  
  > Introduce regresión regularizada (aunque no usa RidgeClassifier directamente).

- *Statistical Learning* – Stanford/MIT (freely available)  
  > Curso hermano de los libros ISLR/ESL, excelente para entender ridge y OvR.

- *StatQuest with Josh Starmer* – YouTube  
  > Explicaciones súper claras sobre regularización, ridge y regresión lineal.

### **Documentación oficial**
- [scikit-learn: RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)  
  > Parámetros, uso, ejemplos y notas técnicas del modelo.

- [scikit-learn: Ridge regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)  
  > Explica su base matemática como regresión regularizada.

> **Consejo**: para dominar `RidgeClassifier`, primero comprende la regresión ridge como regresión continua, luego aplica su lógica a clasificación. Eso te dará una visión completa de su poder y limitaciones.

---
