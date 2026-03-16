import cv2
import numpy as np
import io
from PIL import Image
import math

# Load the logic
from app import calcular_escala_inteligente

# Create a dummy image
img = np.zeros((4000, 3000, 3), dtype=np.uint8)
# Add some drawing to have variance
cv2.rectangle(img, (1000, 1000), (2000, 2000), (255, 255, 255), -1)

ancho_cm = 15.0

print(f"Original shape: {img.shape}")
ancho_px_target, alto_px_target, ppi_actual = calcular_escala_inteligente(img, ancho_cm)
print(f"Target dims: {ancho_px_target}x{alto_px_target}")

if abs(ancho_px_target - img.shape[1]) > 50:
    img_procesada = cv2.resize(img, (ancho_px_target, alto_px_target), interpolation=cv2.INTER_CUBIC)
else:
    img_procesada = img.copy()

print(f"Processed shape: {img_procesada.shape}, actual PPI: {ppi_actual}")

# k-means test
# Z = np.float32(img_procesada.reshape((-1, 3)))
# _, label, center = cv2.kmeans(Z, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
# print("K-means finished")
