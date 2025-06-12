from fourier_mellin import Transform
from toikka2025label.utils.image_registration import (
    ImageRegistrationPreprocessingConfig,
)
import time
import pickle


class OptimalTransformChainsCache:
    def __init__(self):
        self.cache = {}

    def addFeed(self, feed, transforms):
        def convertDictToTransform(t_dict):
            return (
                Transform(
                    t_dict["x"],
                    t_dict["y"],
                    t_dict["scale"],
                    t_dict["rotation"],
                    t_dict["response"],
                )
                if t_dict is not None
                else None
            )

        self.cache[feed] = dict(
            (k2, convertDictToTransform(t)) for k2, t in transforms.items()
        )

    def save(self, filename_pickle, *, info=None):
        if info is None:
            info = f"Saved on {int(time.time())}."
        save_dict = {
            "info": info,
            "cache": self.cache,
        }
        with open(filename_pickle, "wb") as f:
            pickle.dump(save_dict, f)

    def load(self, filename_pickle):
        def convertToTransform(transform_dict):
            return (
                Transform(
                    transform_dict["x"],
                    transform_dict["y"],
                    transform_dict["scale"],
                    transform_dict["rotation"],
                    transform_dict["response"],
                )
                if transform_dict is not None
                else None
            )

        with open(filename_pickle, "rb") as f:
            j = pickle.load(f)
        self.cache = j["cache"]
        for k, v in self.cache.items():
            self.cache[k] = dict(
                (k2, convertToTransform(t_dict)) for k2, t_dict in v.items()
            )

    def __getitem__(self, idx):
        assert isinstance(idx, str)
        return self.cache[idx]


class OptimalTransformChains:
    def __init__(self, rejection_threshold, chain_penalty):
        self.rejection_threshold = rejection_threshold
        self.chain_penalty = chain_penalty
        self.cache = OptimalTransformChainsCache()

    def findOptimalTransformsForFeed(
        self,
        feed,
        transform_matrix,
        feed_target,
        imreg_config: ImageRegistrationPreprocessingConfig,
        *,
        save_to_cache=True,
    ):
        if feed in self.cache.cache:
            return self.cache[feed]

        print(f"Processing optimal chains for {feed}.")
        nodeCount = len(transform_matrix)
        graph = dict(
            (i, list(range(0, i)) + list(range(i + 1, nodeCount)))
            for i in range(nodeCount)
        )

        paths = self._findOptimalPaths(transform_matrix, graph, feed_target)
        graph = self._createGraphFromPaths(paths, feed_target)
        transformsPatch = self._calculateTransforms(
            transform_matrix, graph, feed_target
        )
        transforms = self._correctTransformScale(transformsPatch, imreg_config)
        if save_to_cache:
            self.cache.addFeed(feed, transforms)
        return transforms

    def save(self, filename_json, *, info=None):
        self.cache.save(filename_json, info=info)

    def load(self, filename_json):
        self.cache.load(filename_json)

    def _correctTransformScale(self, transforms, imreg_config):
        resolution = imreg_config.imgSize
        resolution_low = imreg_config.getProcessedResolution()
        transforms2 = {}

        for k, (t, _) in transforms.items():
            pxScaler = imreg_config.cropScale * resolution[0] / (resolution_low[0])
            x = t.x() * pxScaler
            y = t.y() * pxScaler
            scale = t.scale()
            rotation = t.rotation()
            transforms2[k] = Transform(x, y, scale, rotation, t.response())
        return transforms2

    def _calculateTransforms(self, transform_matrix, graph, feed_target):
        def _getTransform(j, i):
            if j > i:
                if j not in transform_matrix or i not in transform_matrix[j]:
                    return None
                return transform_matrix[j][i]
            else:
                if i not in transform_matrix or j not in transform_matrix[i]:
                    return None
                return transform_matrix[i][j].get_inverse()

        def _traverse(parent, nodes):
            for node, children in nodes.items():
                transformRef = transformsToReference[parent][0]
                transformParent = _getTransform(node, parent)
                if transformParent is not None:
                    totalTransform = transformParent * transformRef
                    chainLength = transformsToReference[parent][1] + 1

                    transformsToReference[node] = (totalTransform, chainLength)
                _traverse(node, children)

        transformsToReference = {feed_target: (Transform(), 0)}
        _traverse(feed_target, graph[feed_target])
        return transformsToReference

    def _createGraphFromPaths(self, paths, feed_target):
        def getChildren(parent):
            children = []
            for child, (target, _) in paths.items():
                if target == parent:
                    children.append(child)
            return children

        def traverse(node):
            return dict(
                (child, traverse(child)) for child in getChildren(node) if child != node
            )

        children = traverse(feed_target)
        return {feed_target: children}

    def _findOptimalPaths(self, transform_matrix, graph, feed_target):
        def getScore(a, b):
            b, a = sorted([a, b])
            try:
                t = transform_matrix[a][b]
            except KeyError:
                t = None
            cost = t.response() if t is not None else 0.0
            return min(1.0, max(1e-4, cost))

        def getCombinedScore(score1, score2):
            return score1 * score2 * (1 - 1e-6) * self.chain_penalty

        currentBest = dict()
        for k in graph.keys():
            currentBest[k] = (feed_target, getScore(k, feed_target))

        changed = set(list(graph.keys()))
        while len(changed):
            prevChanged, changed = changed, set()
            for base in graph.keys():
                for target in prevChanged:
                    _, bestScore = currentBest[base]
                    _, score = currentBest[target]
                    directScore = getScore(base, target)
                    newScore = getCombinedScore(directScore, score)
                    if newScore > bestScore:
                        currentBest[base] = (target, newScore)
                        changed.update({base})
        return currentBest


def find_optimal_transforms(feed, label_idx, transform_matrix, imreg_config, chains):
    try:
        for k, v in transform_matrix.items():
            for k2, t_dict in v.items():
                transform_matrix[k][k2] = Transform(
                    t_dict["x"],
                    t_dict["y"],
                    t_dict["scale"],
                    t_dict["rotation"],
                    t_dict["response"],
                )
        _ = chains.findOptimalTransformsForFeed(
            feed, transform_matrix, label_idx, imreg_config
        )
        return True, chains.cache.cache
    except Exception as e:
        print(f"Error processing feed {feed}", e)
        return False, chains.cache.cache
