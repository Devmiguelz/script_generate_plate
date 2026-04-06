# 🚗 Plate Augmentation — Placas Vehiculares Colombianas

Script standalone de **Data Augmentation** para generar datasets de entrenamiento
a partir de fotos reales de placas vehiculares colombianas.

---

## 📋 Requisitos

- Python 3.10+
- Windows / macOS / Linux

---

## ⚙️ Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno virtual
venv\Scripts\Activate.ps1        # Windows PowerShell
source venv/bin/activate          # macOS / Linux

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 📁 Estructura del proyecto

```
generate_plate/
├── augment.py          ← Script principal
├── requirements.txt    ← Dependencias
├── README.md           ← Este archivo
├── fotos/              ← TUS FOTOS ORIGINALES aquí
│   ├── foto001.jpg
│   ├── foto002.jpg
│   └── ...
└── dataset/            ← Imágenes generadas (se crea automáticamente)
    ├── plate_foto001_orig.jpg
    ├── plate_foto001_v0001_a1b2c3.jpg
    └── ...
```

---

## 🚀 Uso

### Comando básico

```bash
python augment.py --input ./fotos --output ./dataset --total 4000 --quality 90
```

### Todos los argumentos

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--input` | Carpeta con las fotos originales | `./input` |
| `--output` | Carpeta donde se guarda el dataset generado | `./output` |
| `--total` | Cantidad total de imágenes a generar | `2000` |
| `--include-original` | Incluir las fotos originales en el output | `True` |
| `--quality` | Calidad JPEG del output (60–100) | `90` |
| `--prefix` | Prefijo para los nombres de archivo | `plate` |

### Ejemplos

```bash
# Dataset de 2000 imágenes
python augment.py --input ./fotos --output ./dataset --total 2000

# Dataset de 4000 imágenes con calidad alta
python augment.py --input ./fotos --output ./dataset --total 4000 --quality 95

# Dataset de 6000 imágenes sin incluir originales
python augment.py --input ./fotos --output ./dataset --total 6000 --include-original false
```

---

## 📊 ¿Cuántas fotos necesito?

| Fotos reales | Total recomendado | Calidad del modelo |
|-------------|------------------|--------------------|
| 50–100 | 2000 | Básico |
| 100–150 | 3000 | Bueno |
| 150–250 | 4000–5000 | Muy bueno |
| 300+ | 6000+ | Profesional |

> **Tip:** La variedad importa más que la cantidad. Asegúrate de tener
> fotos de día, noche, lluvia, distintos ángulos y diferentes tipos de vehículos.

---

## 🎨 Pipelines de Augmentation

El script aplica 3 niveles de transformación aleatoriamente:

### 🟢 Light (40%) — Condiciones normales
- Rotación leve (±8°)
- Brillo y contraste suave
- Ruido gaussiano leve
- Blur / Sharpen
- Zoom aleatorio

### 🟡 Medium (40%) — Condiciones de campo
- Rotación media (±15°) + perspectiva
- Brillo agresivo + escala de grises
- Ruido medio
- Motion blur / Defocus
- Sombras y niebla leve
- Compresión JPEG simulada

### 🔴 Hard (20%) — Condiciones difíciles
- Rotación fuerte (±20°) + distorsión
- Brillo extremo / ecualización
- Ruido agresivo
- Motion blur fuerte
- Lluvia, niebla densa, sombras múltiples
- Downscale (simula cámara baja resolución)

---

## 📦 Formatos de entrada soportados

`.jpg` `.jpeg` `.png` `.webp` `.bmp`

---

## 📤 Formato de salida

Todas las imágenes se guardan como **JPEG** con el siguiente naming:

```
plate_<nombre_original>_orig.jpg       ← copia del original
plate_<nombre_original>_v0001_<uid>.jpg ← variante aumentada
```

---

## ✅ Ejemplo de salida en consola

```
20:07:32  INFO  ============================================================
20:07:32  INFO    Imagenes base      : 204
20:07:32  INFO    Variantes por foto : 19
20:07:32  INFO    Total esperado     : 4080
20:07:32  INFO    Incluir originales : True
20:07:32  INFO    Calidad JPEG       : 90%
20:07:32  INFO    Carpeta output     : C:\...\dataset
20:07:32  INFO  ============================================================
Procesando: 100%|████████████████| 204/204 [04:21<00:00]
20:11:53  INFO    Generadas   : 4080
20:11:53  INFO    Errores     : 0
20:11:53  INFO    Guardadas en: C:\...\dataset
20:11:53  INFO  ============================================================
```

---

## 🔧 Solución de problemas

### Warnings de Albumentations
Si ves warnings como `Argument(s) 'var_limit' are not valid`, asegúrate
de usar la versión más reciente del script que ya tiene los parámetros corregidos.

### Error `rain_type`
Actualiza al script más reciente. El parámetro `rain_type=None` fue
reemplazado por `rain_type='drizzle'` en versiones nuevas de Albumentations.

### No se encontraron imágenes
Verifica que tus fotos estén dentro de la carpeta indicada en `--input`
y que tengan una extensión soportada (`.jpg`, `.png`, etc.).

---

## 📌 Próximos pasos

Una vez generado el dataset puedes usarlo para entrenar un modelo **YOLOv8**
de detección de placas colombianas.

1. Anotar el dataset con [Roboflow](https://roboflow.com) o [LabelImg](https://github.com/HumanSignal/labelImg)
2. Exportar en formato YOLO
3. Entrenar con `yolo train`
4. Integrar el modelo entrenado al backend FastAPI de OCR

---

## 📄 Licencia

Uso interno — Proyecto OCR Placas Vehiculares Colombianas.