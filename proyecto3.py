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
    Muestra una imagen con Matplotlib (convierte BGR a RGB si aplica).
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

    def match(self, imgA, imgB, tag=None):
        kpA, desA = self.get_feats(imgA)
        kpB, desB = self.get_feats(imgB)
        # Mostrar keypoints
        if tag is not None:
            show_image(cv2.drawKeypoints(imgA, kpA, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), f'Keypoints A {tag}')
            show_image(cv2.drawKeypoints(imgB, kpB, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), f'Keypoints B {tag}')
        raw = self.matcher.knnMatch(desA, desB, k=2)
        good = [m for m, n in raw if m.distance < RATIO_TEST * n.distance]
        if tag is not None:
            show_image(cv2.drawMatches(imgA, kpA, imgB, kpB, good, None, flags=2),
                       f'Matches pre RANSAC {tag}')
        if len(good) < 4:
            print(f'No suficientes matches para {tag}')
            return None
        ptsA = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        ptsB = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, REPROJ_THRESH)
        if tag is not None:
            inliers = [good[i] for i in range(len(good)) if mask.ravel()[i]]
            show_image(cv2.drawMatches(imgA, kpA, imgB, kpB, inliers, None, flags=2),
                       f'Inliers {tag}')
        return H


class Stitcher:
    """
    Construye una panorámica generando una homografía acumulada
    desde la imagen central a cada foto, luego warpea y mezcla
    todas sobre un lienzo global con "feather blending".
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

        mid = len(self.images)//2
        self.base  = self.images[mid]
        self.right = self.images[mid+1:]
        self.left  = list(reversed(self.images[:mid]))
        self.matcher = Matcher()

    def stitch(self):
        # 1) Calcular homografías desde la base a cada imagen
        Hs = [np.eye(3)]
        H_acc = np.eye(3)
        prev = self.base
        for i, img in enumerate(self.right):
            H_rel = self.matcher.match(prev, img, tag=f'D{i}')
            H_acc = H_acc.dot(H_rel)
            Hs.append(H_acc.copy())
            prev = img
        H_acc = np.eye(3)
        prev = self.base
        for i, img in enumerate(self.left):
            H_rel = self.matcher.match(prev, img, tag=f'I{i}')
            H_acc = H_acc.dot(H_rel)
            Hs.append(H_acc.copy())
            prev = img

        # 2) Determinar tamaño global del lienzo
        all_corners = []
        imgs = [self.base] + self.right + self.left
        for im, H in zip(imgs, Hs):
            h,w = im.shape[:2]
            corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            warped = cv2.perspectiveTransform(corners, H)
            all_corners.append(warped)
        all_pts = np.vstack(all_corners)
        xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float64)
        canvas_size = (xmax-xmin, ymax-ymin)

        # 3) Warpear todas las imágenes al lienzo y mostrar
        warped_imgs = []
        masks       = []
        for idx, (im, H) in enumerate(zip(imgs, Hs)):
            wim = cv2.warpPerspective(im, T.dot(H), canvas_size)
            msk = cv2.warpPerspective(
                np.ones(im.shape[:2], np.uint8)*255,
                T.dot(H), canvas_size)
            show_image(wim, f'Warp imagen {idx}')
            show_image(msk*255, f'Mascara {idx}')
            warped_imgs.append(wim)
            masks.append(msk)

        # 4) Feather blending global, paso a paso
        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), np.float32)
        weight = np.zeros((canvas_size[1], canvas_size[0]), np.float32)
        for idx, (wim, msk) in enumerate(zip(warped_imgs, masks)):
            dt = cv2.distanceTransform((msk>0).astype(np.uint8), cv2.DIST_L2, 5)
            wgt = dt / (dt.max()+1e-6)
            show_image(wgt, f'Peso distancia {idx}')
            w3  = wgt[:,:,None]
            canvas += wim.astype(np.float32) * w3
            weight += wgt
        final = (canvas / (weight[:,:,None]+1e-6))
        final = np.clip(final, 0, 255).astype(np.uint8)


        return final


if __name__ == '__main__':
    img_dir = os.path.join(os.path.dirname(__file__), 'imagenes')
    stitcher = Stitcher(img_dir)
    pano = stitcher.stitch()
    show_image(pano, 'Panorámica Final')
