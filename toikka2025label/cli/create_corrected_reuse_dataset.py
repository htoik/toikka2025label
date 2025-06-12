import cv2
import fourier_mellin
from toikka2025label.utils.optimal_paths import (
    OptimalTransformChainsCache,
)
from toikka2025label.utils.image_registration import (
    ImageRegistrationPreprocessingConfig,
)


def main():
    import argparse
    import yaml
    import os

    parser = argparse.ArgumentParser(description="Create Corrected Reuse Dataset CLI")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset config YAML."
    )
    parser.add_argument(
        "--optimal-paths-cache",
        type=str,
        required=True,
        help="Path to save the optimal paths cache.",
    )
    args = parser.parse_args()

    dataset_config = yaml.safe_load(open(args.dataset, "r"))
    dataset_path = dataset_config["path"]

    optimal_paths_cache = OptimalTransformChainsCache()
    optimal_paths_cache.load(args.optimal_paths_cache)

    for feed in optimal_paths_cache.cache.keys():
        feed_path = f"{dataset_path}/feeds/{feed}"
        print(f"Processing feed: {feed_path}")
        # feed_config = dataset_config["feeds"][feed]

        img_fps = [os.path.join(feed_path, img) for img in os.listdir(feed_path)]
        imgs = [cv2.imread(fp, cv2.IMREAD_GRAYSCALE) for fp in img_fps]

        imreg_config = ImageRegistrationPreprocessingConfig(
            imgSize=imgs[0].shape[:2][::-1],
            cropScale=0.8,
            maxDimension=1024,
            scaleDownFactor=2,
        )
        imgs = imreg_config.preprocessImages(imgs)

        # transformed = fourier_mellin.get_transformed(img, transform)
        optimal_transforms = optimal_paths_cache[feed]
        corrected_imgs = []
        for img, (transform, _) in zip(imgs, optimal_transforms.items()):
            transformed = fourier_mellin.get_transformed(img, transform)
            corrected_imgs.append(transformed)
        corrected_feed_path = os.path.join(dataset_path, "corrected_feeds", feed)
        os.makedirs(corrected_feed_path, exist_ok=True)
        for i, corrected_img in enumerate(corrected_imgs):
            corrected_img_path = os.path.join(
                corrected_feed_path, f"corrected_{i:09d}.png"
            )
            cv2.imwrite(corrected_img_path, corrected_img)
    print(f"Corrected images saved to {os.path.join(dataset_path, 'corrected_feeds')}")


if __name__ == "__main__":
    main()
