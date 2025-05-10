import os
import sys
import glob
import cv2
import numpy as np

class Matcher:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, img1, img2, direction=None):
        set1 = self.getFeatures(img1)
        set2 = self.getFeatures(img2)
        print(f"Matching direction: {direction}")
        matches = self.bf.knnMatch(
            set2['des'],
            set1['des'],
            k=2
        )
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append((m.trainIdx, m.queryIdx))
        if len(good) > 4:
            pts2 = np.float32([set2['kp'][j].pt for (_, j) in good]).reshape(-1,1,2)
            pts1 = np.float32([set1['kp'][i].pt for (i, _) in good]).reshape(-1,1,2)
            H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
            return H
        return None

    def getFeatures(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}


class Stitch:
    def __init__(self, image_paths):
        self.matcher = Matcher()
        self.images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: could not read {path}")
                continue
            img = cv2.resize(img, (480, 320))
            self.images.append(img)
        self.count = len(self.images)
        if self.count == 0:
            raise ValueError("No valid images to stitch.")
        self.left_list = []
        self.right_list = []
        self.prepare_lists()

    def prepare_lists(self):
        print(f"Total images: {self.count}")
        centerIdx = self.count // 2
        print(f"Center index: {centerIdx}")
        for idx, img in enumerate(self.images):
            if idx <= centerIdx:
                self.left_list.append(img)
            else:
                self.right_list.append(img)
        print("Prepared left and right lists.")

    def leftshift(self):
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.matcher.match(a, b, 'left')
            if H is None:
                continue
            xh = np.linalg.inv(H)
            f1 = xh.dot(np.array([0, 0, 1]))
            f1 /= f1[-1]
            xh[0, -1] += abs(int(f1[0]))
            xh[1, -1] += abs(int(f1[1]))
            ds = xh.dot(np.array([a.shape[1], a.shape[0], 1]))
            ds /= ds[-1]
            dsize = (int(ds[0]) + abs(int(f1[0])), int(ds[1]) + abs(int(f1[1])))
            tmp = cv2.warpPerspective(a, xh, dsize)
            tmp[abs(int(f1[1])):b.shape[0]+abs(int(f1[1])), abs(int(f1[0])):b.shape[1]+abs(int(f1[0]))] = b
            a = tmp
        self.leftImage = a
        print("Left shift complete.")

    def rightshift(self):
        for img in self.right_list:
            H = self.matcher.match(self.leftImage, img, 'right')
            if H is None:
                continue
            txyz = H.dot(np.array([img.shape[1], img.shape[0], 1]))
            txyz /= txyz[-1]
            dsize = (int(txyz[0]) + self.leftImage.shape[1], int(txyz[1]) + self.leftImage.shape[0])
            warped = cv2.warpPerspective(img, H, dsize)
            self.leftImage = self.mix_and_match(self.leftImage, warped)
        print("Right shift complete.")

    def mix_and_match(self, base, warped):
        h, w = base.shape[:2]
        for y in range(h):
            for x in range(w):
                if np.array_equal(warped[y, x], [0, 0, 0]):
                    warped[y, x] = base[y, x]
        return warped


if __name__ == '__main__':
    image_dir = os.path.join(os.path.dirname(__file__), 'imagenes')
    exts = ['*.jpg', '*.jpeg', '*.png']
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths.sort()
    if not paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    stitcher = Stitch(paths)
    stitcher.leftshift()
    stitcher.rightshift()

    out = 'panorama_result.jpg'
    cv2.imwrite(out, stitcher.leftImage)
    print(f"Panorama saved to {out}")
