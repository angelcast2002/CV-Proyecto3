import os
import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parámetros globales
# -----------------------------
RESIZE_DIM = (480, 320)
RATIO_TEST = 0.75
REPROJ_THRESH = 4.0
DETECTOR_TYPE = 'ORB'   # Elige entre 'ORB', 'SIFT'
NFEATURES = 2000       # Número de características para ORB

# Mostrar imágenes con matplotlib
def show_image(img, title=None):
    if img is None:
        return
    if len(img.shape) == 3 and img.shape[2] == 3:
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display = img
    plt.figure(figsize=(10,6))
    if title: plt.title(title)
    plt.axis('off')
    plt.imshow(display)
    plt.show()

class Matcher:
    def __init__(self, detector_type=DETECTOR_TYPE, nfeatures=NFEATURES):
        # Configurar detector y medida de distancia
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
            norm = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(nfeatures=nfeatures)
            norm = cv2.NORM_HAMMING
        self.matcher = cv2.BFMatcher(norm, crossCheck=False)

    def get_feats(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des

    def match_and_show(self, imgA, imgB, tag):
        # Extraer y visualizar keypoints
        kpA, desA = self.get_feats(imgA)
        kpB, desB = self.get_feats(imgB)
        show_image(cv2.drawKeypoints(imgA, kpA, None,
                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), f'Keypoints {tag} A')
        show_image(cv2.drawKeypoints(imgB, kpB, None,
                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), f'Keypoints {tag} B')
        # Matching con ratio test
        raw = self.matcher.knnMatch(desA, desB, k=2)
        good = [m for m,n in raw if m.distance < RATIO_TEST * n.distance]
        show_image(cv2.drawMatches(imgA, kpA, imgB, kpB, good, None, flags=2),
                   f'Matches pre RANSAC {tag}')
        if len(good) < 4:
            print(f'No suficientes matches para {tag}')
            return None
        ptsA = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        ptsB = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, REPROJ_THRESH)
        inliers = [good[i] for i in range(len(good)) if mask.ravel()[i]]
        show_image(cv2.drawMatches(imgA, kpA, imgB, kpB, inliers, None, flags=2),
                   f'Inliers {tag}')
        return H

class Stitcher:
    def __init__(self, paths):
        self.images = []
        for p in sorted(paths):
            img = cv2.imread(p)
            if img is not None:
                self.images.append(cv2.resize(img, RESIZE_DIM))
        if not self.images:
            raise RuntimeError('No hay imágenes válidas en el directorio')
        mid = len(self.images) // 2
        self.base = self.images[mid]
        self.left = list(reversed(self.images[:mid]))
        self.right = self.images[mid+1:]
        self.matcher = Matcher()

    def warp_blend(self, panoramica, img, H, tag):
        # Calcular nuevo lienzo
        hP, wP = panoramica.shape[:2]
        hI, wI = img.shape[:2]
        corners = np.float32([[0,0],[wI,0],[wI,hI],[0,hI]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_pts = np.vstack((np.float32([[0,0],[wP,0],[wP,hP],[0,hP]]).reshape(-1,1,2), warped_corners))
        xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float64)
        size = (xmax-xmin, ymax-ymin)
        # Warpear y mostrar
        pan_w = cv2.warpPerspective(panoramica, T, size)
        img_w = cv2.warpPerspective(img, T.dot(H), size)
        show_image(pan_w, f'Warp panoramica {tag}')
        show_image(img_w, f'Warp imagen {tag}')
        # Blending simple (pixel > 0)
        mask = img_w > 0
        merged = pan_w.copy()
        merged[mask] = img_w[mask]
        show_image(merged, f'Blend {tag}')
        return merged

    def stitch(self):
        # Comenzar con la imagen base
        panoramica = self.base.copy()
        # Extender a la derecha
        for idx, img in enumerate(self.right):
            H = self.matcher.match_and_show(panoramica, img, f'D{idx}')
            if H is not None:
                panoramica = self.warp_blend(panoramica, img, H, f'D{idx}')
        # Extender a la izquierda
        for idx, img in enumerate(self.left):
            H = self.matcher.match_and_show(panoramica, img, f'I{idx}')
            if H is not None:
                panoramica = self.warp_blend(panoramica, img, H, f'I{idx}')
        return panoramica

if __name__ == '__main__':
    dir_img = os.path.join(os.path.dirname(__file__), 'imagenes')
    paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        paths.extend(glob.glob(os.path.join(dir_img, ext)))
    stitcher = Stitcher(paths)
    panoramica = stitcher.stitch()
    show_image(panoramica, 'Panoramica Final')
