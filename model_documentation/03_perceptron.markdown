# <center><b>Perceptron</b></center>

---

## <b>¿Qué es?</b>

El **Perceptrón** es un modelo clásico de clasificación supervisada que busca separar dos clases mediante una **frontera de decisión lineal**. Fue propuesto por Frank Rosenblatt en 1958 y es considerado el precursor de las redes neuronales.

En su versión moderna (`sklearn.linear_model.Perceptron`), se entrena mediante **descenso de gradiente estocástico**, admite **regularización L2** y puede resolver tareas de clasificación **binaria o multiclase** (One-vs-Rest).

Aunque su simplicidad limita su uso en problemas complejos, sigue siendo útil en datos **linealmente separables** o como modelo base por su rapidez y bajo costo computacional.

---

## <b>Formulación matemática</b>

El **Perceptrón** busca aprender una frontera de decisión **lineal** para separar dos clases etiquetadas como $y \in \{-1, +1\}$, usando una combinación lineal de los atributos de entrada.

Dado un conjunto de entrenamiento con $n$ ejemplos y $d$ características:

- $X = \{x^{(1)}, \dots, x^{(n)}\} \subset \mathbb{R}^d$
- $y = \{y^{(1)}, \dots, y^{(n)}\} \subset \{-1, +1\}$

el modelo estima un **vector de pesos** $w \in \mathbb{R}^d$ y un **sesgo (intercepto)** $b \in \mathbb{R}$ que definen una función de predicción lineal:

$$
f(x) = \text{sign}(w^\top x + b)
$$

### **Proceso de aprendizaje**

El Perceptrón **no minimiza una función de pérdida explícita** como otros modelos. En lugar de eso, **ajusta los pesos solo cuando comete un error de clasificación**. Su actualización es:

Para cada observación mal clasificada $(x^{(i)}, y^{(i)})$:

$$
\begin{aligned}
w &\leftarrow w + \eta \cdot y^{(i)} \cdot x^{(i)} \\
b &\leftarrow b + \eta \cdot y^{(i)}
\end{aligned}
$$

- $\eta$ es la **tasa de aprendizaje** (learning rate).
- El proceso se repite durante varias **épocas** sobre los datos.

### **Importante: implementación moderna (`scikit-learn`)**

En `scikit-learn`, `Perceptron` es una **versión generalizada** que:

- Utiliza **descenso de gradiente estocástico (SGD)** en lugar de la regla pura original.
- Puede incorporar **regularización L2**.
- Puede verse como un modelo que **minimiza una pérdida tipo hinge modificada**, muy similar a un SVM lineal sin margen blando.

Aunque conceptualmente no se define una función de pérdida, su comportamiento es **equivalente a minimizar el número de errores**, o en su versión moderna, una forma suavizada del *hinge loss*:

$$
\text{hinge}(y, z) = \max(0, 1 - y z)
$$

Donde $z = w^\top x + b$.

### **Predicción final**

La decisión sobre la clase predicha se hace evaluando el signo de la combinación lineal:

$$
\hat{y} = \text{sign}(w^\top x + b)
$$

> En resumen: El perceptrón aprende iterativamente actualizando sus pesos solo cuando se equivoca, generando una frontera lineal que busca separar las clases. En `scikit-learn`, se ha adaptado para aprovechar SGD y regularización, haciéndolo más robusto para uso práctico.

---

## <b>Supuestos</b>

Para que el **Perceptrón** funcione correctamente y produzca resultados útiles, se deben tener en cuenta los siguientes puntos:

- **Separabilidad lineal aproximada**  
  > El Perceptrón clásico **solo converge si los datos son linealmente separables**. En caso contrario, el algoritmo puede no encontrar una solución estable.  
  > *Por eso `scikit-learn` usa una cantidad máxima de iteraciones (`max_iter`) y permite regularización.*

- **No requiere normalidad, homocedasticidad ni independencia de variables**  
  > A diferencia de otros modelos estadísticos clásicos, el Perceptrón **no impone supuestos sobre la distribución** de los predictores o del error.

- **Escalado de las variables es altamente recomendable**  
  > Debido a que se basa en combinaciones lineales y actualizaciones graduales, **el rango de las variables puede afectar drásticamente la convergencia y estabilidad**.  
  > *Se recomienda usar `StandardScaler` o `MinMaxScaler` antes de entrenar.*

- **Sensibilidad a outliers**  
  > Como el Perceptrón no minimiza una pérdida robusta, **puede ser afectado por observaciones extremas**, especialmente si están mal etiquetadas.  
  > *No tiene tolerancia incorporada al ruido como los modelos con márgenes suaves (como SVM).*

- **Datos independientes** (deseable, no obligatorio)  
  > Aunque el algoritmo no asume independencia explícitamente, **los datos correlacionados (por ejemplo, series temporales) pueden afectar el aprendizaje**.

> El Perceptrón es un modelo simple y rápido, pero su desempeño **depende críticamente de la calidad y separabilidad de los datos**. Es más efectivo cuando las clases se pueden dividir con una línea (o hiperplano) razonablemente clara.
>
> ---
>
> ## <b>Interpretación del modelo</b>

Aunque el **Perceptrón** no estima probabilidades ni una función de pérdida explícita, **sí genera un modelo lineal con coeficientes interpretables**, cuya estructura puede analizarse como sigue:

### **Coeficientes (pesos)**

Los **pesos** $w = [w_1, w_2, ..., w_p]$ y el **intercepto** $b$ definen una función lineal:

$$
f(x) = w^\top x + b
$$

- Si $f(x) > 0$, se predice una clase (por ejemplo, $+1$); si $f(x) < 0$, se predice la otra (por ejemplo, $-1$).
- La **magnitud** de cada $w_j$ indica la **importancia del predictor $x_j$** en la decisión.
- El **signo de $w_j$** indica la dirección de la influencia:
  - Si $w_j > 0$, valores altos de $x_j$ empujan la predicción hacia la clase positiva.
  - Si $w_j < 0$, hacia la clase negativa.

> No se puede interpretar como odds ni probabilidades, solo **como impacto direccional sobre la predicción lineal**.

### **Intercepto $b$**

El **intercepto** (o sesgo) desplaza la frontera de decisión hacia arriba o abajo en el espacio de características.

- Un $b > 0$ favorece predicciones positivas.
- Un $b < 0$ favorece predicciones negativas.

### **Frontera de decisión**

El Perceptrón genera una **hiperplano de decisión lineal**:

$$
w^\top x + b = 0
$$

- Esta es la frontera que separa ambas clases en el espacio de entrada.
- Los puntos más alejados de esta línea tienen predicciones más "seguras", aunque **no se puede cuantificar la confianza** como con la regresión logística.

### **¿Y si hay más de 2 clases?**

El modelo implementa una estrategia **One-vs-Rest (OvR)** de forma interna:

- Se entrena un modelo binario por clase.
- Cada modelo predice su clase frente a las demás.
- La clase elegida es la que tenga el valor más alto de $w^\top x + b$.

### **En resumen:**

| Elemento         | Significado                                                   |
|------------------|---------------------------------------------------------------|
| $w_j$            | Peso de la variable $x_j$ en la decisión                      |
| Signo de $w_j$   | Dirección de la influencia (a favor/en contra de una clase)   |
| $b$              | Desplazamiento de la frontera de decisión                     |
| $w^\top x + b$   | Score lineal (no es probabilidad)                             |
| Frontera         | Hiperplano que divide las clases                              |
| Salida final     | Clase con score mayor (en binario: sign del score)            |

---

## <b>Implementación en `scikit-learn`</b>

La clase `Perceptron` se encuentra en el módulo `sklearn.linear_model`. Aquí mostramos cómo implementarla y ajustar los hiperparámetros clave:

```python
from sklearn.linear_model import Perceptron

model = Perceptron(
    penalty=None,
    alpha=0.0001,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-3,
    shuffle=True,
    verbose=0,
    eta0=1.0,
    random_state=42,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    class_weight=None,
    warm_start=False
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
```

---

## <b>Parámetros cruciales</b>

Para ajustar correctamente un modelo Perceptrón y obtener buenos resultados, es fundamental entender estos hiperparámetros clave:

### **penalty**

- Especifica el tipo de regularización:
  - `'l2'` → penaliza coeficientes grandes suavemente.
  - `'l1'` → fuerza algunos coeficientes a ser cero (sparse).
  - `'elasticnet'` → combinación de ambas.
  - `None` (por defecto) → sin regularización.
- Ayuda a prevenir **overfitting**, especialmente con muchas variables o ruido.
- Solo funciona si se especifica un valor distinto de `None`.

### **alpha**

- Controla la **fuerza de la regularización**. Actúa como un coeficiente multiplicador sobre la penalización.
- Funciona como el $\lambda$ clásico en otros modelos.

> Aunque el Perceptrón clásico no tiene regularización, esta versión moderna sí la incluye para mayor estabilidad y control.  
> Valores pequeños → regularización débil. Valores grandes → penalización fuerte que simplifica el modelo.

### **eta0**

- Tasa de aprendizaje **fija** (no adaptativa).
- Afecta directamente el tamaño del paso en la regla de actualización de pesos.

> Un valor muy grande puede hacer que el modelo oscile y no converja.  
> Un valor muy pequeño puede hacer que tarde mucho o se quede atascado.

### **max_iter**

- Número máximo de **épocas**: es decir, pasadas completas sobre el conjunto de entrenamiento.

> Cuanto más grande sea el conjunto de datos, más iteraciones puede requerir para aprender correctamente.

### **tol**

- Criterio de tolerancia para detener el entrenamiento antes de `max_iter`.
- Si la mejora en el error no supera `tol`, se considera que el modelo ha convergido.

> Si se quiere forzar a que siempre entrene `max_iter`, se puede usar `tol=None`.

### **early_stopping + validation_fraction + n_iter_no_change**

- `early_stopping`: habilita el uso de una partición de validación interna para **detener el entrenamiento cuando no hay mejora**.
- `validation_fraction`: proporción del set de entrenamiento reservada como validación.
- `n_iter_no_change`: número de épocas consecutivas sin mejora para activar el stop.

> Útil para evitar overfitting en datasets grandes.  
> Desactiva validación cruzada explícita cuando está activo.

### **class_weight**

- Ajusta la importancia de cada clase.  
- Opciones:
  - `'balanced'`: compensa clases minoritarias automáticamente.
  - `{0: 1, 1: 3}`: pesos personalizados.

> Muy recomendable usarlo si hay **desbalance de clases**, para mejorar recall o F1-score en la clase minoritaria.

### **fit_intercept**

- Si `True` (default), el modelo aprende un **término de sesgo** $b$.
- Si los datos están centrados, puede ponerse `False`.

### **shuffle**

- Si mezcla los datos antes de cada época.
- Recomendado dejar en `True` para mejorar estabilidad del aprendizaje estocástico.

### **Recomendaciones finales**

- Realiza **búsqueda de hiperparámetros cruzada** (`GridSearchCV`) para `alpha`, `eta0` y `penalty`.
- Usa `StandardScaler` antes de entrenar: mejora rendimiento y estabilidad.

---

## <b>Validaciones Numéricas Internas</b>

Cuando ejecutas `model.fit(X, y)` con `Perceptron`, internamente el modelo realiza un **entrenamiento secuencial** basado en errores. Aquí te explico el proceso paso a paso:

### **Inicialización de coeficientes**

- Se crean los pesos $w \in \mathbb{R}^p$ y el sesgo $b$, inicializados a ceros (o aleatorios si `warm_start=True`).
- Si `fit_intercept=True`, también se añade el término de sesgo $b$.

### **Entrenamiento iterativo (por épocas)**

El entrenamiento recorre varias veces el conjunto de datos (`max_iter` épocas). En cada época:

1. Se mezclan los datos si `shuffle=True`.
2. Se analiza cada ejemplo $(x^{(i)}, y^{(i)})$:
   - Se calcula la predicción:
     $$
     \hat{y}^{(i)} = \text{sign}(w^\top x^{(i)} + b)
     $$
   - Si la predicción **es incorrecta**, se actualizan los pesos:
     $$
     w \leftarrow w + \eta_0 \cdot y^{(i)} \cdot x^{(i)} \\
     b \leftarrow b + \eta_0 \cdot y^{(i)}
     $$
   - Si `penalty` está activado, se aplica regularización:
     $$
     w \leftarrow w - \eta_0 \cdot \alpha \cdot w
     $$

> Nota: Esto se repite solo para errores. El Perceptrón **no actualiza si acierta** la clase.

### **Parada por tolerancia (`tol`)**

- Después de cada época, el modelo evalúa el desempeño.
- Si no mejora el score en validación durante `n_iter_no_change` épocas consecutivas, **se detiene** si `early_stopping=True`.

### **Salidas del modelo**

Después del entrenamiento, quedan disponibles:

- `.coef_`: pesos $w$ aprendidos
- `.intercept_`: sesgo $b$ (si se usa)
- `.n_iter_`: número real de épocas ejecutadas
- `.classes_`: clases aprendidas

> Aunque el modelo es simple, permite entender de forma clara **cómo aprende una frontera de decisión** usando errores y actualizaciones directas.

---

## <b>Casos de uso</b>

Aunque hoy en día existen modelos más avanzados, el **Perceptrón sigue siendo útil** en ciertos contextos específicos por su simplicidad, rapidez y bajo costo computacional. Aquí detallamos **cuándo y por qué** elegirlo:

### **Clasificación rápida en datos linealmente separables**

El Perceptrón converge **solo si los datos son separables linealmente**, lo que lo hace ideal en estos casos:

- Detección binaria de patrones simples.
- Tareas donde el margen entre clases es amplio.
- Prototipado rápido en datasets perfectamente etiquetados.

> Si los datos no son separables linealmente, **no garantiza convergencia** (por eso se usa `max_iter`).

### **Ambientes con recursos limitados**

Debido a su bajo consumo de memoria y cómputo, es excelente para:

- Dispositivos embebidos o edge computing.
- Aplicaciones con restricciones de latencia o memoria.
- Algoritmos de bajo costo computacional para inferencia rápida.

### **Educación y análisis teórico**

El Perceptrón es uno de los **mejores modelos para enseñar**:

- Fundamentos del aprendizaje supervisado.
- Descenso de gradiente estocástico (SGD).
- Concepto de margen y separación lineal.

> También es un punto de entrada para entender **redes neuronales**.

### **Clasificadores en línea (streaming)**

Aunque no está diseñado explícitamente para streaming, su implementación secuencial lo hace compatible con:

- Datos que llegan en tiempo real.
- Entrenamiento progresivo (`partial_fit`) en lotes pequeños.

### **Problemas con muchas features pero pocas muestras**

Gracias a su simplicidad, el Perceptrón:

- No necesita invertir mucho tiempo en ajuste de hiperparámetros.
- Puede funcionar razonablemente bien cuando hay **alta dimensionalidad** y **pocas observaciones**, siempre que las clases sean separables.

> El Perceptrón **no estima probabilidades**, ni maneja relaciones no lineales. En esos casos, modelos como `LogisticRegression`, `SVM`, `MLPClassifier` o árboles pueden ser mejores alternativas.

---

## <b>Profundización matemática</b>

### **Función de decisión**

El Perceptrón busca un **hiperplano** que separe las clases:

$$
f(x) = \text{sign}(w^\top x + b)
$$

- $w \in \mathbb{R}^d$: vector de pesos.
- $b$: sesgo.
- La función `sign` devuelve $+1$ o $-1$, según en qué lado del hiperplano esté la muestra.

### **Regla de actualización clásica**

Cada vez que se encuentra un ejemplo mal clasificado:

- Si $y^{(i)} (w^\top x^{(i)} + b) \leq 0$, se actualiza:

$$
\begin{align*}
w &\leftarrow w + \eta \cdot y^{(i)} \cdot x^{(i)} \\
b &\leftarrow b + \eta \cdot y^{(i)}
\end{align*}
$$

Donde $\eta$ es la tasa de aprendizaje (en `sklearn`, `eta0`).

> Este método **no busca minimizar una función de pérdida**, sino corregir errores de clasificación directamente.

### **Versión moderna en `scikit-learn`**

El `Perceptron` en `scikit-learn` está basado en **SGDClassifier** con:

- Pérdida `loss='perceptron'`
- `learning_rate='constant'`
- `eta0` fijo
- Sin penalización por defecto (`penalty=None`)

Esto permite agregar:

- **Regularización** ($\ell_1$, $\ell_2$ o `elasticnet`)
- **Criterios de parada**
- **Early stopping**
- **Tamaño máximo de iteraciones**

### **No hay función de pérdida suave**

A diferencia de `LogisticRegression` (con `log_loss`) o `SVM` (con pérdida tipo hinge), el Perceptrón clásico no tiene una función continua o derivable. La versión moderna implementa una **pérdida cero-uno escalonada**:

$$
\text{PerceptronLoss}(y, \hat{y}) = 
\begin{cases}
0 & \text{si } y = \hat{y} \\
1 & \text{si } y \ne \hat{y}
\end{cases}
$$

Lo cual **no es diferenciable**, pero `sklearn` lo entrena usando un esquema adaptado de SGD.

### **Comparación con otros modelos**

| Modelo              | Función de pérdida              | Convergencia asegurada |
|---------------------|----------------------------------|--------------------------|
| Perceptrón clásico  | Ninguna (actualización directa) | Solo si hay separación lineal |
| Regresión logística | Log-loss                        | Sí, en convexos          |
| SVM lineal          | Hinge loss                      | Sí, en convexos          |

> *El Perceptrón es un caso particular de modelos lineales entrenados con SGD, sin pérdida diferenciable ni márgenes explícitos como SVM.*

---

## <b>Recursos para profundizar</b>

### **Libros**

- **Understanding Machine Learning** – Shalev-Shwartz & Ben-David  
  > Contiene una excelente explicación matemática del Perceptrón clásico y sus garantías de convergencia.

- **The Elements of Statistical Learning** – Hastie, Tibshirani, Friedman  
  > Aborda el Perceptrón desde la perspectiva de métodos lineales y aprendizaje supervisado.

- **Pattern Recognition and Machine Learning** – Christopher Bishop  
  > Incluye el Perceptrón dentro de un marco probabilístico más amplio (aunque con menor énfasis práctico).

- **Deep Learning** – Goodfellow, Bengio, Courville  
  > Explica cómo el Perceptrón fue la base histórica de las redes neuronales modernas.

### **Cursos**

- **Machine Learning – Andrew Ng (Coursera)**  
  > Introduce el Perceptrón en el contexto de clasificación lineal y redes neuronales primitivas.

- **CS231n (Stanford)**  
  > Aunque se enfoca en redes neuronales profundas, cubre brevemente el Perceptrón como base histórica.

- **StatQuest (YouTube)**  
  > Explicaciones gráficas claras del Perceptrón y cómo se actualiza paso a paso.

### **Documentación oficial**

- [`sklearn.linear_model.Perceptron`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)  
  > Detalles de implementación, hiperparámetros y ejemplos de uso.

### **Papers históricos y modernos**

- **"The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" – Frank Rosenblatt (1958)**  
  > Artículo original del creador del Perceptrón.

- **"A short introduction to boosting" – Freund & Schapire**  
  > Discute cómo el Perceptrón motivó modelos como AdaBoost (al mejorar modelos débiles).

> *Aunque simple, el Perceptrón fue el primer paso hacia redes neuronales profundas. Entenderlo bien te da una base sólida para todo el aprendizaje automático.*

---
