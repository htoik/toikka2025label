import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from fourier_mellin import Transform, FourierMellin, FourierMellinWithReference


class ImageRegistrationPreprocessingConfig:
    def __init__(self, imgSize, *, cropScale, maxDimension, scaleDownFactor):
        self.cropScale = cropScale
        self.maxDimension = maxDimension
        self.scaleDownFactor = scaleDownFactor
        self.imgSize = imgSize
        self.frameSize = self._getFrameSize()
        self.cropSize = (maxDimension * cropScale, maxDimension * cropScale)
        self.cropSize = tuple(int(x) for x in self.cropSize)

    def getProcessedResolution(self):
        return tuple(x // self.scaleDownFactor + 1 for x in self.cropSize)

    def _getFrameSize(self):
        if self.imgSize[0] < self.imgSize[1]:
            return (
                self.maxDimension,
                int(self.imgSize[1] / self.imgSize[0] * self.maxDimension),
            )
        else:
            return (
                int(self.imgSize[0] / self.imgSize[1] * self.maxDimension),
                self.maxDimension,
            )

    def preprocessImages(self, imgs):
        imgs = [self._preprocessImage(img) for img in imgs]
        return imgs

    def _preprocessImage(self, img):
        img = cv2.resize(img, self.frameSize, cv2.INTER_CUBIC)
        w, h = self.frameSize
        wi, hi = self.cropSize
        x = (w - wi) // 2
        y = (h - hi) // 2
        return img[
            y : y + hi : self.scaleDownFactor, x : x + wi : self.scaleDownFactor
        ].copy()


class ImageRegistration:
    def __init__(self, config: ImageRegistrationPreprocessingConfig, *, maxThreads):
        self.maxThreads = maxThreads
        self.config = config

    def preprocessImages(self, imgs):
        imgs = self.config.preprocessImages(imgs)
        return imgs

    def registerOne(self, img1, img2, idx) -> Transform:
        fm = FourierMellin(*img1.shape[:2][::-1])
        _, transform = fm.register_image(img1, img2)
        return transform.to_dict(), idx

    def register(self, imgs, idxs: list[tuple[int, int]]):
        imgs = self.config.preprocessImages(imgs)
        results = [None for _ in range(len(idxs))]
        with ProcessPoolExecutor(self.maxThreads) as executor:
            future = {
                executor.submit(self.registerOne, imgs[i], imgs[j], idx)
                for idx, (i, j) in enumerate(idxs)
            }
            for f in as_completed(future):
                transform, idx = f.result()
                results[idx] = transform
        return results

    def _registerRow(self, imgs, reference, targetIdxs):
        row = []
        fm = FourierMellinWithReference(*self.config.getProcessedResolution())
        fm.set_reference(reference, -1)
        for targetIdx in targetIdxs:
            target = imgs[targetIdx]
            _, transform = fm.register_image(target)
            row.append(transform)
        return row

    def _processRow(self, imgs, idx, idxs2):
        reference = imgs[idx]
        targetIdxs = idxs2
        transformsRow = self._registerRow(imgs, reference, targetIdxs)

        transformsRow = [
            t.get_inverse().to_dict() if x > idx else t.to_dict()
            for t, x in zip(transformsRow, idxs2)
        ]
        return idx, transformsRow

    def registerImagesPaired(
        self, imgs, idxs1: list[int], idxs2: list[int], *, skipPreprocessing=True
    ):
        if not skipPreprocessing:
            imgs = self.config.preprocessImages(imgs)
        results = {}
        with ProcessPoolExecutor(self.maxThreads) as executor:
            future_to_ref_i = {
                executor.submit(self._processRow, imgs, y, idxs2): (y,) for y in idxs1
            }
            for future in as_completed(future_to_ref_i):
                ref_i, row = future.result()
                results[ref_i] = row
        return results
