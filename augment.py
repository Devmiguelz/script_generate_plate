"""
Data Augmentation para Placas Vehiculares Colombianas
======================================================
Script standalone — sin servidor, sin API.

Uso:
    python augment.py --input ./fotos --output ./dataset --total 2000

Argumentos:
    --input             Carpeta con las imágenes originales  (default: ./input)
    --output            Carpeta donde se guardan los resultados (default: ./output)
    --total             Cantidad total de imágenes a generar  (default: 2000)
    --include-original  Incluir las fotos originales en el output (default: True)
    --quality           Calidad JPEG del output 60-100        (default: 90)
    --prefix            Prefijo para los archivos generados   (default: plate)

Ejemplos:
    python augment.py
    python augment.py --input ./mis_fotos --total 4000
    python augment.py --input ./fotos --output ./dataset --total 1000 --quality 85

Instalar dependencias:
    pip install albumentations opencv-python-headless numpy pillow tqdm
"""

import argparse
import shutil
import uuid
import random
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─── Pipelines de Augmentation ───────────────────────────────────────────────

def pipeline_light() -> A.Compose:
    """Suave — condiciones normales de día."""
    return A.Compose([
        A.OneOf([
            A.Rotate(limit=8, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.Affine(shear=(-8, 8), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.8),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=1.0),
        ], p=0.3),
        A.RandomScale(scale_limit=0.15, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
    ])


def pipeline_medium() -> A.Compose:
    """Medio — condiciones de campo, sombras, angulos."""
    return A.Compose([
        A.OneOf([
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
            A.Affine(shear=(-15, 15), scale=(0.85, 1.15), p=1.0),
            A.Perspective(scale=(0.05, 0.12), p=1.0),
        ], p=0.85),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
            A.RandomGamma(gamma_limit=(70, 140), p=1.0),
            A.ToGray(p=1.0),
        ], p=0.75),
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.20), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), p=1.0),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(1, 3), p=1.0),
        ], p=0.45),
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                           num_shadows_limit=(1, 2),
                           shadow_dimension=4, p=1.0),
            A.RandomFog(fog_coef_range=(0.05, 0.2),
                        alpha_coef=0.1, p=1.0),
        ], p=0.35),
        A.ImageCompression(quality_range=(65, 95), p=0.4),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=0.4),
    ])


def pipeline_hard() -> A.Compose:
    """Agresivo — noche, lluvia, desenfoque fuerte."""
    return A.Compose([
        A.OneOf([
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
            A.Affine(shear=(-20, 20), scale=(0.75, 1.25), p=1.0),
            A.Perspective(scale=(0.1, 0.2), p=1.0),
        ], p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
            A.RandomGamma(gamma_limit=(50, 180), p=1.0),
            A.Equalize(p=1.0),
        ], p=0.85),
        A.OneOf([
            A.GaussNoise(std_range=(0.10, 0.40), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.7, 1.3), elementwise=True, p=1.0),
        ], p=0.65),
        A.OneOf([
            A.MotionBlur(blur_limit=(5, 15), p=1.0),
            A.Defocus(radius=(2, 5), p=1.0),
        ], p=0.55),
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                           num_shadows_limit=(1, 3),
                           shadow_dimension=5, p=1.0),
            A.RandomFog(fog_coef_range=(0.15, 0.4),
                        alpha_coef=0.15, p=1.0),
            A.RandomRain(slant_range=(-10, 10),
                         drop_length=8, drop_width=1,
                         drop_color=(180, 180, 180),
                         blur_value=3,
                         brightness_coefficient=0.85,
                         rain_type='drizzle', p=1.0),
        ], p=0.45),
        A.ImageCompression(quality_range=(40, 75), p=0.5),
        A.Downscale(scale_min=0.5, scale_max=0.8, p=0.3),
    ])


# Distribucion: 40% suave / 40% medio / 20% duro
_PIPELINES = [
    (pipeline_light,  0.40),
    (pipeline_medium, 0.40),
    (pipeline_hard,   0.20),
]

def pick_pipeline() -> A.Compose:
    r = random.random()
    acc = 0.0
    for builder, prob in _PIPELINES:
        acc += prob
        if r < acc:
            return builder()
    return pipeline_medium()


# ─── Utilidades de imagen ─────────────────────────────────────────────────────

def load_image(path: Path):
    """Carga imagen BGR, con fallback a PIL para formatos poco comunes."""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"No se pudo cargar {path.name}: {e}")
        return None


def save_image(img: np.ndarray, path: Path, quality: int = 90) -> bool:
    """Guarda imagen BGR como JPEG."""
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            path.write_bytes(buf.tobytes())
            return True
    except Exception as e:
        logger.warning(f"Error guardando {path.name}: {e}")
    return False


def augment(img: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """Aplica el pipeline (trabaja en RGB internamente)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = pipeline(image=rgb)["image"]
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ─── Logica principal ─────────────────────────────────────────────────────────

def run(
    input_dir: Path,
    output_dir: Path,
    total: int,
    include_original: bool,
    quality: int,
    prefix: str,
) -> None:

    # Recopilar imagenes base
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not images:
        logger.error(f"No se encontraron imagenes en '{input_dir}'")
        logger.error(f"Formatos soportados: {', '.join(SUPPORTED_EXTS)}")
        sys.exit(1)

    n_base = len(images)
    variants_per_image = max(1, total // n_base)
    expected = n_base * variants_per_image + (n_base if include_original else 0)

    logger.info("=" * 60)
    logger.info(f"  Imagenes base      : {n_base}")
    logger.info(f"  Variantes por foto : {variants_per_image}")
    logger.info(f"  Total esperado     : {expected}")
    logger.info(f"  Incluir originales : {include_original}")
    logger.info(f"  Calidad JPEG       : {quality}%")
    logger.info(f"  Carpeta output     : {output_dir.resolve()}")
    logger.info("=" * 60)

    # Preparar output — limpiar si ya existe
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    generated = 0
    errors    = 0

    for img_path in tqdm(images, desc="Procesando", unit="foto"):
        img = load_image(img_path)
        if img is None:
            errors += variants_per_image
            continue

        stem = img_path.stem

        # Copiar original si se pidio
        if include_original:
            dest = output_dir / f"{prefix}_{stem}_orig.jpg"
            if save_image(img, dest, quality):
                generated += 1

        # Generar variantes
        for v in range(variants_per_image):
            try:
                aug_img = augment(img, pick_pipeline())
                uid     = uuid.uuid4().hex[:6]
                dest    = output_dir / f"{prefix}_{stem}_v{v:04d}_{uid}.jpg"
                if save_image(aug_img, dest, quality):
                    generated += 1
                else:
                    errors += 1
            except Exception as e:
                logger.warning(f"  Error en {stem} variante {v}: {e}")
                errors += 1

    logger.info("=" * 60)
    logger.info(f"  Generadas   : {generated}")
    logger.info(f"  Errores     : {errors}")
    logger.info(f"  Guardadas en: {output_dir.resolve()}")
    logger.info("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un dataset aumentado de placas vehiculares colombianas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",            type=Path, default=Path("./input"),
                        help="Carpeta con las imagenes originales")
    parser.add_argument("--output",           type=Path, default=Path("./output"),
                        help="Carpeta donde se guardaran las imagenes generadas")
    parser.add_argument("--total",            type=int,  default=2000,
                        help="Cantidad total de imagenes a generar (aprox)")
    parser.add_argument("--include-original", action="store_true", default=True,
                        help="Incluir las fotos originales en el output")
    parser.add_argument("--quality",          type=int,  default=90,
                        help="Calidad JPEG del output (60-100)")
    parser.add_argument("--prefix",           type=str,  default="plate",
                        help="Prefijo para los nombres de archivo generados")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        logger.error(f"La carpeta de entrada no existe: '{args.input}'")
        logger.error("Creala y coloca tus fotos dentro, o usa --input <ruta>")
        sys.exit(1)

    run(
        input_dir        = args.input,
        output_dir       = args.output,
        total            = args.total,
        include_original = args.include_original,
        quality          = args.quality,
        prefix           = args.prefix,
    )