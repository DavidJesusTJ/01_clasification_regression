# <center><b> Regresión Logística </b></center>

---

## <b>¿Qué es?</b>

La **regresión logística** es un modelo que se utiliza para **clasificar casos en dos o más categorías posibles** (por ejemplo, “sí” o “no”, “compra” o “no compra”, “enfermo” o “sano”).

Lo que hace este modelo es tomar un conjunto de características (como edad, ingresos, comportamiento, etc.) y **estimar la probabilidad de que algo ocurra**.

Es especialmente útil cuando queremos responder preguntas como:

- ¿Un cliente dejará de usar el servicio?
- ¿Un paciente tiene una enfermedad?
- ¿Este correo es spam?

Aunque se llama “regresión”, en realidad **no predice valores numéricos continuos**, sino **probabilidades** que luego se pueden usar para tomar decisiones de clasificación.

---

## <b>Formulación Matemática</b>

### **Binaria**

Imagina que queremos predecir si un cliente pagará o no pagará un préstamo.

La regresión logística no predice directamente 0 o 1, sino la probabilidad de que ocurra el evento (por ejemplo, que sí pague).

La fórmula con variables $X = (x_1, ..., x_p)$ es:

$$
P(Y = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{donde } z = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
$$

La función $\sigma(z)$ es la **sigmoide**. Cuanto mayor $z$, más cercana a 1 será la probabilidad.

¿Qué significa eso?

* Tomamos una combinación lineal de las variables (edad, ingresos, etc.) → eso da el valor $z$.
* Pasamos ese $z$ por la función sigmoide $\sigma(z)$, que lo convierte en una probabilidad entre 0 y 1.
* Si la probabilidad es mayor a un umbral (como 0.5), predecimos que sí ocurrirá el evento.

La sigmoide suaviza esa decisión y nos da una curva de probabilidad.

### **Multiclase**

Cuando hay más de dos clases (por ejemplo, si el cliente puede ser de tipo A, B o C), usamos dos enfoques:

* **One-vs-Rest (OvR)**
    * Entrenamos un modelo para cada clase: por ejemplo, Clase A vs. no A, Clase B vs. no B, etc.


* **Softmax (regresión logística multinomial)**:
    * Aquí el modelo predice todas las probabilidades al mismo tiempo, usando esta fórmula:

    $$
    P(y = k \mid X) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{con } z_k = X^\top \beta_k
    $$

    * Esto asegura que todas las probabilidades sumen 1.
    * Cada clase tiene su propio vector de pesos $\beta_k$.

### **Función de Pérdida: `log_loss`**

La **regresión logística** aprende ajustando sus parámetros para **minimizar el error entre las probabilidades que predice y los valores reales**.

Ese error se mide con una función llamada **pérdida logarítmica negativa** (`log_loss`), que penaliza más fuertemente cuando el modelo está **muy seguro y se equivoca**.

**Para el caso binario:**

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

* Esta función:

    - Da una pérdida **baja** si el modelo predice una probabilidad alta cerca del valor real.
    - Da una pérdida **alta** si el modelo predice una probabilidad alta hacia el lado equivocado.
    - Por eso es ideal para clasificación probabilística: **no solo importa acertar, sino qué tan convencido estás al hacerlo**.

**Para el caso multiclase (con softmax):**

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(p_{ik})
$$

* Donde:

    - $y_{ik}$ es 1 si la observación $i$ pertenece a la clase $k$, y 0 en caso contrario.
    - $p_{ik}$ es la probabilidad que el modelo asignó a la clase $k$ para esa observación.


Aquí, el modelo intenta **asignar la mayor probabilidad a la clase correcta**, penalizando cualquier distribución equivocada.

En resumen: `log_loss` no solo mide si acertamos o no, sino **qué tan bien calibradas están nuestras predicciones**. Un modelo que duda con frecuencia (predice 0.51 en lugar de 0.99) tendrá más pérdida, aunque acierte.

---

## <b>Supuestos del Modelo</b>

Para que la regresión logística funcione correctamente y produzca resultados confiables, se deben considerar los siguientes supuestos:

- **Relación lineal** entre las variables independientes y el logit (log-odds) de la variable dependiente.  
  > *No se requiere linealidad con la variable respuesta directamente, sino con su log-odds.*

- **Independencia de las observaciones**.  
  > *Cada fila debe ser independiente. No es adecuado para datos dependientes (como series temporales o datos de panel sin ajustes).*

- **Ausencia de multicolinealidad severa** entre las variables predictoras.  
  > *Se recomienda revisar VIFs o usar técnicas como PCA si hay correlación fuerte entre predictores.*

- **Tamaño de muestra suficiente**.  
  > *Se sugiere tener al menos 10 eventos por cada predictor para evitar overfitting.*

- **Ausencia de outliers influyentes o leverage points excesivos**.  
  > *Los outliers pueden distorsionar los coeficientes. Es recomendable revisar medidas como Cook's Distance o leverage.*

- **No hay errores de medición severos en las variables independientes**.  
  > *Se asume que los predictores son medidos con cierta precisión. El error en X puede afectar la estimación.*

*A diferencia de la regresión lineal, la regresión logística **no asume normalidad ni homocedasticidad** de los residuos.*

---

## <b>Interpretación</b>

Una vez entrenado el modelo de regresión logística, es clave **interpretar correctamente sus salidas**. A continuación, se detallan los elementos que deben analizarse:

### **Coeficientes y log-odds**

Cada coeficiente $\beta_j$ representa el **efecto del predictor $x_j$ sobre el logaritmo del odds** (logit) de que ocurra el evento (por ejemplo, $Y = 1$).

$$
\text{logit}(p) = \log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p
$$

- Si $\beta_j > 0$: a mayor valor de $x_j$, **mayor probabilidad** del evento.
- Si $\beta_j < 0$: a mayor valor de $x_j$, **menor probabilidad** del evento.
- Si $\beta_j = 0$: $x_j$ no tiene efecto sobre la respuesta.

### **Odds y Odds Ratio**

- **Odds**: se refiere a la razón entre la probabilidad de que ocurra el evento y la de que no ocurra:
  $$
  \text{odds} = \frac{p}{1 - p}
  $$

- **Odds Ratio** (OR): se interpreta como el cambio multiplicativo en los odds ante un incremento de 1 unidad en $x_j$:
  $$
  \text{OR}_j = e^{\beta_j}
  $$

> Ejemplo: Si $\beta_1 = 0.7$, entonces $e^{0.7} \approx 2.01$, lo que significa que un aumento de 1 unidad en $x_1$ **duplica los odds** del evento.

### **Probabilidades**

El modelo calcula una probabilidad para cada observación:

$$
p_i = \frac{1}{1 + e^{-z_i}} = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_j x_{ij})}}
$$

Esto se interpreta como la **probabilidad estimada** de que $Y = 1$ dado $X$.

### **Significancia estadística de los coeficientes**

Cada coeficiente $\beta_j$ tiene asociado:

- Un **error estándar**
- Un **valor z**: $\dfrac{\beta_j}{SE(\beta_j)}$
- Un **valor p**: para evaluar si el efecto es significativo

> Un valor p < 0.05 indica que el predictor **tiene un efecto significativo** en el modelo.

### **Desvianza (Deviance)**

La **desvianza** mide el mal ajuste del modelo. Es análoga a la suma de cuadrados de errores en regresión lineal:

- **Desvianza del modelo completo**: 
  $$
  D = -2 \cdot \log(\text{verosimilitud del modelo})
  $$

- **Desvianza nula**: usando solo el intercepto  
- **Desvianza residual**: con todas las variables predictoras

> Un **buen modelo** reduce la desviación residual respecto a la nula.

### **Pseudo R²**

Como no se puede usar el R² tradicional, se emplean **versiones adaptadas**:

- **McFadden's $R^2$**:
  $$
  R^2_{\text{McFadden}} = 1 - \frac{\log L_{\text{modelo}}}{\log L_{\text{nulo}}}
  $$

- **Cox & Snell R²**, **Nagelkerke R²** (ajustado)

> Aunque no equivalen a un R² clásico, **valores más altos indican mejor ajuste relativo**.

### **En resumen:**

| Elemento          | Qué representa                                    |
|-------------------|----------------------------------------------------|
| $\beta_j$         | Efecto de $x_j$ sobre el log-odds                  |
| $e^{\beta_j}$     | Cambio en los odds (odds ratio)                   |
| Desvianza         | Mal ajuste del modelo                             |
| Pseudo $R^2$      | Calidad relativa del ajuste                       |
| p-valor           | Significancia del efecto de cada variable         |

---

## <b>Implementación en `scikit-learn`</b>

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2',
    dual=False,
    tol=1e-4,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=42,
    solver='lbfgs',
    max_iter=100,
    multi_class='auto',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
```

---

## <b>Parámetros Cruciales</b>


A continuación, se explican los hiperparámetros más importantes que afectan directamente el rendimiento, generalización y comportamiento de la regresión logística.

### **penalty — Tipo de regularización**

La regularización **penaliza** los coeficientes del modelo para evitar que crezcan demasiado y se sobreajusten a los datos de entrenamiento.

- `'l2'` → Ridge: penaliza los **cuadrados** de los coeficientes. Tiende a mantenerlos pequeños pero no los lleva a cero.
- `'l1'` → Lasso: penaliza los **valores absolutos**. Puede hacer que algunos coeficientes sean exactamente cero (selección de variables).
- `'elasticnet'`: mezcla entre L1 y L2. Necesita que `l1_ratio` esté definido.
- `'none'`: sin regularización. **Muy peligroso si tienes muchos predictores o poco data.**

> Cuanto más fuerte la regularización, más "conservador" será tu modelo.

### **C — Inverso de la fuerza de regularización**

Este es uno de los más malentendidos. **`C` NO es coste ni penalidad directa. Es el inverso de la regularización**:

- **Valor bajo de `C`** ⇒ regularización fuerte ⇒ el modelo ajusta menos a los datos ⇒ **más generalización**.
- **Valor alto de `C`** ⇒ regularización débil ⇒ el modelo trata de ajustar más exacto al entrenamiento ⇒ **mayor riesgo de sobreajuste**.

> `C = 1 / λ` donde λ es la fuerza de regularización.

> En práctica: prueba varios valores con validación cruzada (por ejemplo, `C` en [0.01, 0.1, 1, 10, 100]).

### **solver — Algoritmo de optimización**

El solver define **cómo se entrena el modelo numéricamente**. Afecta:

- Velocidad
- Estabilidad
- Soporte para diferentes penalizaciones y multiclase

| Solver        | L1 | L2 | Multiclase  | Escala bien con grandes datos |
| ------------- |----|----|-------------|-------------------------------|
| `'liblinear'` | Sí  | Sí  | No (OvR)     | Lento en muchos datos         |
| `'lbfgs'`     | No  | Sí  | Sí (softmax) | Rápido y estable            |
| `'newton-cg'` | No  | Sí  | Sí           | Bien para softmax             |
| `'sag'`       | No  | Sí  | Sí           | Muy rápido con muchas filas   |
| `'saga'`      | Sí  | Sí  | Sí           | Ideal para datos grandes + L1/L2 |

> Solo `'liblinear'` y `'saga'` soportan `penalty='l1'` o `elasticnet`.

### **multi_class — Estrategia para multiclase**

Solo relevante si tienes más de dos clases.

- `'auto'`: selecciona `'ovr'` o `'multinomial'` según el `solver`.
- `'ovr'`: entrena un modelo por clase vs. el resto (One-vs-Rest).
- `'multinomial'`: entrena un solo modelo conjunto con softmax.  
    Mejor si las clases se superponen mucho.

> Usa `'multinomial'` + `solver='lbfgs'` o `'saga'` para mejores resultados.

### **class_weight — Control del desbalance de clases**

Cuando una clase ocurre mucho más que otra (por ejemplo, 90% vs 10%), el modelo puede **ignorar** la clase minoritaria.

- `None`: todas las clases tienen el mismo peso.
- `'balanced'`: ajusta los pesos automáticamente según la frecuencia de cada clase.
- `{0: w0, 1: w1}`: pesos personalizados si sabes cuánto penalizar cada clase.

> En clasificación desbalanceada, `class_weight='balanced'` puede **mejorar mucho el recall** en la clase minoritaria.

### **Otros parámetros útiles pero no críticos**

Si bien no afectan directamente el rendimiento predictivo, estos pueden ayudarte en casos específicos:

- `max_iter`: Aumenta si el solver no converge.
- `tol`: Reduce si quieres mayor precisión en la convergencia.
- `fit_intercept`: Generalmente debe ser `True`.

**Resumen gráfico mental:**

| Parámetro     | Afecta...                   | Cuándo ajustarlo                        |
|---------------|-----------------------------|-----------------------------------------|
| `penalty`     | Qué coeficientes se penalizan | Para evitar overfitting o seleccionar variables |
| `C`           | Cuánta regularización aplicar | Cuando hay over/underfitting            |
| `solver`      | Cómo se entrena el modelo     | Según tamaño de datos y penalización     |
| `multi_class` | Estrategia para multiclase    | Cuando tienes más de dos clases         |
| `class_weight`| Cómo trata el desbalance      | Siempre que tengas clases desbalanceadas |

---

## <b>Validaciones Numéricas Internas</b>

Cuando llamas al método `.fit()` del modelo de regresión logística en `scikit-learn`, se inicia un proceso interno de **optimización matemática** para encontrar los mejores coeficientes.

### **¿Qué significa "entrenar" el modelo?**

Significa **encontrar los valores óptimos de los coeficientes** ($\beta_0, \beta_1, ..., \beta_p$) que **minimizan la función de pérdida** definida por el modelo.

### **¿Qué función se minimiza?**

Se minimiza la **pérdida logarítmica negativa** (`log_loss`), también conocida como **verosimilitud negativa**:

- Para clasificación binaria:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

- Si hay regularización (según `penalty` y `C`), se agrega un término extra:

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda R(\beta)
$$

Donde $R(\beta)$ puede ser:
- $\sum \beta_j^2$ si `penalty='l2'`
- $\sum |\beta_j|$ si `penalty='l1'`

### **¿Qué hace internamente `.fit()`?**

1. **Inicializa** los coeficientes con valores pequeños (aleatorios o ceros).
2. **Calcula gradientes** de la función de pérdida respecto a cada coeficiente.
3. **Itera usando el solver elegido** (como `'lbfgs'`, `'liblinear'`, etc.) para actualizar los coeficientes en la dirección que minimiza la pérdida.
4. **Detiene el entrenamiento** cuando:
   - Se alcanza el número máximo de iteraciones (`max_iter`), o
   - La mejora es menor que la tolerancia (`tol`)

### **¿Qué devuelve al final?**

Después del ajuste, puedes obtener:

- `model.coef_`: los coeficientes $\beta_1, \dots, \beta_p$
- `model.intercept_`: el intercepto $\beta_0$
- `model.n_iter_`: número de iteraciones realizadas
- `model.classes_`: las clases aprendidas

### **Importante:**

- Si el modelo **no converge**, puedes:
  - Aumentar `max_iter`
  - Usar un solver más estable (`'lbfgs'`, `'saga'`)
  - Revisar escalado de variables

- Si estás usando **regularización**, el efecto dependerá de `C`, `penalty` y el tipo de solver.

### **En resumen**

Entrenar una regresión logística significa resolver un **problema de optimización numérica**, donde el objetivo es **maximizar la verosimilitud** (o minimizar la pérdida logarítmica) y encontrar los coeficientes que mejor explican la relación entre $X$ e $Y$.

---

## <b>Casos de uso</b>

Aunque hoy en día se prefieren modelos complejos como XGBoost, LightGBM o redes neuronales, la regresión logística sigue siendo **la mejor elección** en ciertos escenarios.

### **Diagnóstico médico (enfermo vs. sano)**

- Necesitas **interpretabilidad clara**: saber qué variables aumentan el riesgo.
- La regresión logística permite obtener **odds ratios**, ideales para publicaciones clínicas.

> Los médicos necesitan confianza en el modelo, no solo predicción.

### **Crédito y scoring financiero**

- **Regulaciones legales** exigen modelos interpretables.
- Permite explicar por qué se aprueba o rechaza un crédito.
- Es **fácil de auditar** por entes reguladores.

> En banca, transparencia > precisión ciega.

### **Detección de fraude (primera etapa)**

- Útil como **modelo base rápido** en sistemas antifraude.
- Funciona bien cuando las **relaciones son lineales o log-odds**.
- Se puede escalar y mantener sin complejidad computacional.

> Luego puede reemplazarse por XGBoost, pero primero se parte por modelos simples.

### **Clasificación de correos (spam vs. no spam)**

- Si el volumen de datos es muy alto, un modelo simple y rápido puede ser suficiente.
- Puede ser entrenado en tiempo real sin necesidad de GPUs ni infraestructura pesada.

> Perfecto para soluciones ligeras y reentrenamiento frecuente.

### **Predicción de abandono (churn)**

- Se busca **entender las causas del abandono** (interpretabilidad).
- El equipo de negocio puede accionar directamente sobre los coeficientes del modelo.

> Saber que "aumentar llamadas reduce churn" es más útil que un AUC de 0.94 sin explicaciones.

### **¿Cuándo *NO* usar regresión logística?**

- Cuando las relaciones entre variables y el target **no son lineales**.
- Cuando hay **mucha interacción o complejidad no capturada por logit**.
- Cuando **la precisión máxima es crítica** (por ejemplo, en detección de fraude en tiempo real con millones de eventos).

### **Conclusión**

> La regresión logística no es anticuada: es **la opción correcta cuando necesitas transparencia, velocidad, estabilidad y explicaciones claras**.

---

## <b>Profundización matemática</b>

Esta sección está dirigida a quienes desean comprender el **fundamento matemático detrás del entrenamiento** de la regresión logística, más allá de la implementación.

### **Derivada de la función de pérdida (gradiente)**

Para clasificación binaria, la función de pérdida por observación es:

$$
\mathcal{L}_i = -\left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

donde:

$$
p_i = \frac{1}{1 + e^{-z_i}}, \quad z_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}
$$

El **gradiente respecto a un coeficiente $\beta_j$** es:

$$
\frac{\partial \mathcal{L}}{\partial \beta_j} = \sum_{i=1}^{N} (p_i - y_i) x_{ij}
$$

> Esto tiene un significado intuitivo: el gradiente es proporcional al **error de predicción multiplicado por el input correspondiente**.

### **Regularización L2 (Ridge)**

Agrega un término cuadrático a la función de pérdida:

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\beta\|_2^2 = \mathcal{L} + \lambda \sum_{j=1}^{p} \beta_j^2
$$

Esto **reduce la magnitud** de los coeficientes, previene overfitting y favorece soluciones más estables numéricamente.

> 🔧 En `scikit-learn`, se usa $\lambda = \dfrac{1}{C}$, donde `C` es el hiperparámetro de regularización.

### **Regularización L1 (Lasso)**

Agrega la **suma de valores absolutos** en lugar del cuadrado:

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\beta\|_1 = \mathcal{L} + \lambda \sum_{j=1}^{p} |\beta_j|
$$

Esto favorece **soluciones sparse**, es decir, con coeficientes exactamente cero.  
Muy útil para **selección automática de variables**.

> La función no es diferenciable en cero, por lo que requiere métodos especiales como `'liblinear'` o `'saga'` para su optimización.

### **Gradiente completo con regularización**

- Para **L2**:

$$
\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \beta_j} = \sum_{i=1}^{N} (p_i - y_i) x_{ij} + 2\lambda \beta_j
$$

- Para **L1**:  
El gradiente se reemplaza por un **subgradiente**, ya que $|\beta_j|$ no es diferenciable en cero:

$$
\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \beta_j} = \sum_{i=1}^{N} (p_i - y_i) x_{ij} + \lambda \cdot \text{sign}(\beta_j)
$$

donde:

$$
\text{sign}(\beta_j) =
\begin{cases}
1 & \text{si } \beta_j > 0 \\
-1 & \text{si } \beta_j < 0 \\
\in [-1, 1] & \text{si } \beta_j = 0
\end{cases}
$$

### **Nota sobre convergencia y métodos numéricos**

- La regresión logística **no tiene solución cerrada** como la regresión lineal.
- El proceso de ajuste implica **algoritmos iterativos**:
    - `'lbfgs'`, `'newton-cg'`: usan gradiente + Hessiano (segunda derivada).
    - `'liblinear'`, `'saga'`: más eficientes para penalización L1.

### **Conclusión**

> El corazón de la regresión logística es la **minimización de una función convexa**, y su entrenamiento requiere comprender derivadas, regularización y gradientes. Esta base te permite extenderte a modelos más complejos como redes neuronales o XGBoost con confianza.

---

## <b>Recursos para profundizar</b>

**Libros**  
- *The Elements of Statistical Learning* – Hastie, Tibshirani, Friedman  
- *An Introduction to Statistical Learning* – James et al.  
- *Pattern Recognition and Machine Learning* – Bishop  

**Cursos**  
- Andrew Ng – Coursera ML  
- MIT – Statistical Learning  
- FastAI o YouTube (StatQuest)

**Documentación oficial**  
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---
