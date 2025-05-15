import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parámetros globales
# -----------------------------
RESIZE_DIM    = (480, 320)
RATIO_TEST    = 0.75
REPROJ_THRESH = 4.0
DETECTOR_TYPE = 'ORB' 
NFEATURES     = 2000  


def show_image(img, title=None):
    """
    Muestra una imagen con Matplotlib
    """
    if img is None:
        return
    disp = img
    if img.ndim == 3 and img.shape[2] == 3:
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.imshow(disp)
    plt.show()


class Matcher:
    """
    Detecta y empareja características entre dos imágenes.
    """
    def __init__(self, detector_type=DETECTOR_TYPE, nfeatures=NFEATURES):
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.norm     = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(nfeatures=nfeatures)
            self.norm     = cv2.NORM_HAMMING
        self.matcher = cv2.BFMatcher(self.norm, crossCheck=False)

    def get_feats(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(gray, None)

    def match_and_show(self, imgA, imgB, tag):
        # Extraer y visualizar keypoints
        kpA, desA = self.get_feats(imgA)
        kpB, desB = self.get_feats(imgB)
        #show_image(cv2.drawKeypoints(imgA, kpA, None,
         #            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), f'Keypoints {tag} A')
        #show_image(cv2.drawKeypoints(imgB, kpB, None,
        #             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), f'Keypoints {tag} B')
        # Matching con ratio test
        raw = self.matcher.knnMatch(desA, desB, k=2)
        good = [m for m,n in raw if m.distance < RATIO_TEST * n.distance]
        #show_image(cv2.drawMatches(imgA, kpA, imgB, kpB, good, None, flags=2),
         #          f'Matches pre RANSAC {tag}')
        if len(good) < 4:
            print(f'No suficientes matches para {tag}')
            return None
        ptsA = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        ptsB = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, REPROJ_THRESH)
        inliers = [good[i] for i in range(len(good)) if mask.ravel()[i]]
        #show_image(cv2.drawMatches(imgA, kpA, imgB, kpB, inliers, None, flags=2),
        #           f'Inliers {tag}')
        return H


class Stitcher:
    """
    Construye una panorámica empalmando imágenes alrededor de una base, usando feather blending.
    """
    def __init__(self, img_dir):
        # Carga y redimensiona
        paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        self.images = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                self.images.append(cv2.resize(img, RESIZE_DIM))
        if not self.images:
            raise RuntimeError(f"No hay imágenes válidas en {img_dir}")

        # Imagen base (centro) y listas para izquierda/derecha
        mid = len(self.images) // 2
        self.base  = self.images[mid]
        self.left  = list(reversed(self.images[:mid]))
        self.right = self.images[mid+1:]

        self.matcher = Matcher()

    def warp_blend(self, pano, img, H, tag):
        """
        Warpea y mezcla "feather" dos imágenes.
        """
        hP, wP = pano.shape[:2]
        hI, wI = img.shape[:2]
        # esquinas de la imagen a warpear
        corners = np.float32([[0,0],[wI,0],[wI,hI],[0,hI]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, H)

        pano_corners = np.float32([[0,0],[wP,0],[wP,hP],[0,hP]]).reshape(-1,1,2)
        all_pts = np.vstack((pano_corners, warped_corners))
        xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float64)
        size = (xmax-xmin, ymax-ymin)

        pano_w = cv2.warpPerspective(pano, T, size)
        img_w  = cv2.warpPerspective(img, T.dot(H), size)

        #show_image(pano_w, f'Warp panoramica {tag}')
        #show_image(img_w, f'Warp imagen {tag}')

        # máscaras binarias (1 donde hay píxel)
        maskP = (pano_w > 0).any(axis=2).astype(np.uint8)
        maskI = (img_w  > 0).any(axis=2).astype(np.uint8)

        # transformadas de distancia
        dtP = cv2.distanceTransform(maskP, cv2.DIST_L2, 5)
        dtI = cv2.distanceTransform(maskI, cv2.DIST_L2, 5)
        sumDT = dtP + dtI
        sumDT[sumDT == 0] = 1
        wP = dtP / sumDT
        wI = dtI / sumDT

        # expandir pesos a 3 canales
        wP = wP[:, :, np.newaxis]
        wI = wI[:, :, np.newaxis]

        # mezcla ponderada
        blended = (pano_w.astype(np.float32) * wP + img_w.astype(np.float32) * wI)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        #show_image(blended, f'Blend {tag}')
        return blended

    def stitch(self):
        """
        Empalma la panorámica: base -> derecha -> izquierda
        """
        pano = self.base.copy()
        # derecha
        for idx, img in enumerate(self.right):
            H = self.matcher.match_and_show(pano, img, f'D{idx}')
            if H is None:
                print(f"No suficientes matches derecha {idx}")
                continue
            pano = self.warp_blend(pano, img, H, f'D{idx}')
        # izquierda
        for idx, img in enumerate(self.left):
            H = self.matcher.match_and_show(pano, img, f'I{idx}')
            if H is None:
                print(f"No suficientes matches izquierda {idx}")
                continue
            pano = self.warp_blend(pano, img, H, f'I{idx}')

        return pano


if __name__ == '__main__':
    img_dir = os.path.join(os.path.dirname(__file__), 'imagenes')
    stitcher = Stitcher(img_dir)
    pano = stitcher.stitch()
    show_image(pano, 'Panorámica Final')
