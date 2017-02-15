import unittest
from super_resolution import *
import cv2


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_extract_patch_list(self):
        img_path = "../data/urban_hr/img_006_SRF_4_HR.png"
        img = cv2.imread(img_path)

        patch_list = extract_patch_list(img, (41, 41), (21, 21))

        for patch in patch_list:
            cv2.imshow("patch", patch)
            cv2.waitKey(0)

    def test_load_imgs(self):
        img_list = load_img_list_and_extract_patch_list("../data/urban_hr", size=(41, 41), stride=(21, 21))

        for patch in img_list:
            cv2.imshow("patch", patch)
            cv2.waitKey(0)

    def test_blur_img_list(self):
        img_list = load_img_list_and_extract_patch_list("../data/urban_hr", size=(41, 41), stride=(21, 21))

        blur_list = blur_img_list(img_list, scale=2)
        for patch in blur_list:
            cv2.imshow("patch", patch)
            cv2.waitKey(0)
