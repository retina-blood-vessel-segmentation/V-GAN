# Author: Gorana Gojic <crvenpaka@gmail.com>
# Last updated: 26.04.2020.

import argparse
import numpy as np

from pathlib import Path
from skimage import filters
from cv2 import imwrite, imread

import utils

DEFAULT_PROBABILITY_MAP_EXTENSION = ".tif"
DEFAULT_MASK_EXTENSION = ".tif"


def vessels_from_probability_maps(
    probability_maps_dir,
    masks_dir,
    output_dir,
    probability_map_extension = DEFAULT_PROBABILITY_MAP_EXTENSION,
    mask_extension = DEFAULT_MASK_EXTENSION
):

    vessel_probability_maps_files = utils.all_files_under(probability_maps_dir, extension=probability_map_extension)
    mask_files = utils.all_files_under(masks_dir, extension=mask_extension)

    print(len(vessel_probability_maps_files))
    print(len(mask_files))

    # probability maps and masks must be named so that loaded maps and masks are correspondent, otherwise this
    # will not work (fix it one day)
    for map_mask_pair in zip(vessel_probability_maps_files, mask_files):
        vessel_probability_map = imread(map_mask_pair[0], 0)
        vessel_probability_array = np.array(vessel_probability_map, dtype=np.uint8).flatten() # row major flatten

        mask = imread(map_mask_pair[1], 0)
        mask_array = np.array(mask, dtype=np.uint8).flatten()

        try:
            thresh = filters.threshold_otsu(vessel_probability_array[mask_array != 0])
        except ValueError:
            thresh = 0 # thresholding a black image

        result = vessel_probability_array <= thresh
        result = np.array(result, dtype=np.uint8)
        result = np.reshape(result, vessel_probability_map.shape)
        result = 255 - result * 255  # from [0,1] to [0, 255]

        result_path = Path(output_dir) / ((Path(map_mask_pair[0]).stem) + ".png")
        imwrite(
            str(result_path),
            result
        )

    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--probability_maps_dir", type=str, default=None)
    parser.add_argument("--probability_map_extension", type=str, default=DEFAULT_PROBABILITY_MAP_EXTENSION)
    parser.add_argument("--masks_dir", type=str, default=None)
    parser.add_argument("--mask_extension", type=str, default=DEFAULT_MASK_EXTENSION)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    probability_maps_dir        = args.probability_maps_dir
    probability_map_extension   = args.probability_map_extension
    masks_dir                   = args.masks_dir
    mask_extension              = args.mask_extension
    output_dir                  = args.output_dir

    # generate vessel segmentation maps from probability maps
    vessels_from_probability_maps(
        probability_maps_dir = probability_maps_dir,
        probability_map_extension = probability_map_extension,
        masks_dir = masks_dir,
        mask_extension = mask_extension,
        output_dir = output_dir
    )


main()










