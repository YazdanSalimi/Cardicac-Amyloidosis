import os
from glob import glob
from natsort import os_sorted
###########
lsit_tb_original_urls = "list of path to your nifti files, Please put each nifti file in a separate folder"
model_path_on_your_device = r"path to trained model folder containing classificatio and segmentation folders"
target = r"path where you want to save the results. I need this folder to be shared with me."

yazdan.ATTR.external_classify_whole(lsit_tb_original_urls = lsit_tb_original_urls,
                                    target = target, 
                                    model_path_on_your_device =model_path_on_your_device,
                                    )
