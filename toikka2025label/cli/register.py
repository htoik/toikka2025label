import os
import cv2
import yaml
from toikka2025label.utils.image_registration import (
    ImageRegistrationPreprocessingConfig,
    ImageRegistration,
)
from toikka2025label.utils.utils import float_range


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Image Registration CLI")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset config YAML."
    )
    args = parser.parse_args()

    dataset_config = yaml.safe_load(open(args.dataset, "r"))
    dataset_path = dataset_config["path"]

    results = {}

    for feed in dataset_config["feeds"]:
        feed_path = os.path.join(dataset_path, "feeds", feed)
        print(f"Processing feed: {feed_path}")
        img_fps = [os.path.join(feed_path, img) for img in os.listdir(feed_path)]
        imgs = [cv2.imread(fp, cv2.IMREAD_GRAYSCALE) for fp in img_fps]

        imreg_config = ImageRegistrationPreprocessingConfig(
            imgSize=imgs[0].shape[:2][::-1],
            cropScale=0.8,
            maxDimension=1024,
            scaleDownFactor=2,
        )
        imreg = ImageRegistration(config=imreg_config, maxThreads=4)
        block_size = 24
        block_skip_distance = 8
        density_decay_factor = 1.35

        frame_idxs = list(range(len(imgs)))
        frame_count = len(frame_idxs)
        blocks = (frame_count + block_size - 1) // block_size
        feed_results = {}

        print("Preprocessing images.")
        imgs = imreg.preprocessImages(imgs)

        print("Starting image registration.")
        for progress_i, i in enumerate(reversed(range(blocks))):
            print(f"Image registration {progress_i:03d}/{blocks:03d}")
            for j in range(i + 1):
                d = abs(i - j)
                if d > block_skip_distance:
                    continue
                idxs1 = list(
                    float_range(
                        i * block_size,
                        min((i + 1) * block_size, frame_count),
                        density_decay_factor**d,
                    )
                )
                idxs2 = list(
                    float_range(
                        j * block_size,
                        min((j + 1) * block_size, frame_count),
                        density_decay_factor**d,
                    )
                )

                idxs1 = [int(idx) for idx in idxs1]
                idxs2 = [int(idx) for idx in idxs2]

                imgs2 = [imgs[i] for i in idxs1] + [imgs[i] for i in idxs2]
                idxs_local1 = list(range(len(idxs1)))
                idxs_local2 = list(range(len(idxs1), len(idxs1) + len(idxs2)))

                transforms = imreg.registerImagesPaired(imgs2, idxs_local1, idxs_local2)
                for idx1, row in transforms.items():
                    for idx2, transform in zip(idxs2, row):
                        if idx1 not in feed_results:
                            feed_results[idx1] = {}
                        feed_results[idx1][idx2] = transform
        results[feed] = feed_results
    print("Image registration completed.")
    output_path = os.path.join(
        f"./output/registration_results_{dataset_config['name']}.json"
    )
    with open(output_path, "w") as f:
        import json

        json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
