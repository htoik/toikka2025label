import json
import yaml
import cv2
import os
from toikka2025label.utils.image_registration import (
    ImageRegistrationPreprocessingConfig,
)
from toikka2025label.utils.optimal_paths import (
    find_optimal_transforms,
    OptimalTransformChains,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Image Registration CLI")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset config YAML."
    )
    parser.add_argument(
        "--registration-results",
        type=str,
        required=True,
        help="Path to the registration results JSON file.",
    )

    args = parser.parse_args()

    dataset_config = yaml.safe_load(open(args.dataset, "r"))
    dataset_path = dataset_config["path"]

    registration_results = json.load(open(args.registration_results, "r"))
    optimal_transforms = {}

    rejection_threshold = 0.45
    chain_penalty = 0.95

    chains = OptimalTransformChains(rejection_threshold, chain_penalty)

    for feed in dataset_config["feeds"]:
        feed_path = f"{dataset_path}/feeds/{feed}"
        print(f"Processing feed: {feed_path}")
        feed_config = dataset_config["feeds"][feed]
        mask_idx = feed_config["mask_idx"]

        img0_shape = cv2.imread(
            os.path.join(feed_path, os.listdir(feed_path)[0]), cv2.IMREAD_GRAYSCALE
        ).shape[:2][::-1]

        imreg_config = ImageRegistrationPreprocessingConfig(
            imgSize=img0_shape, cropScale=0.8, maxDimension=1024, scaleDownFactor=2
        )

        _, optimal_transforms = find_optimal_transforms(
            feed=feed,
            label_idx=mask_idx,
            transform_matrix=registration_results[feed],
            imreg_config=imreg_config,
            chains=chains,
        )
        chains.cache.addFeed(feed, optimal_transforms)
    chains.cache.save(f"./output/optimal_paths_cache_{dataset_config.name}.json")


if __name__ == "__main__":
    main()
