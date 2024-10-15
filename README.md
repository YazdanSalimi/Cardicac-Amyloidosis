# Cardicac-Amyloidosis
## Automated Cardiac amyloidosis detection and quantification on SPECT scentigraphy images.
## Pipeline Overview
### 1. Cardiac Region Detection: 
  The pipeline first detects cardiac regions using a segmentation model based on the [SwinUnetR](https://github.com/Project-MONAI/research-contributions/tree/main) architecture.
### 2. Image Classification: 
  After segmentation, the pipeline utilizes three pretrained models to classify the images in two main tasks:
- Detection: Identifying relevant features in the images.
- Severity Scoring: Assessing severity based on the Perugini score.
## Download Trained Models
please dowload the [trained models](https://drive.google.com/drive/folders/1eQXQZMW-uIOsw1BoQ880VeEoL3PQCyCV?usp=drive_link) and tell the inference function where you saved those models on your machine.
the [inference instruction](https://github.com/YazdanSalimi/Cardicac-Amyloidosis/blob/main/inference-example.py) is provided. 
To install this repository, simply run:

```bash
pip install git+https://github.com/YazdanSalimi/Cardicac-Amyloidosis.git
```

We welcome any feedback, suggestions, or contributions to improve this project!

for any furtehr question please email me at: [salimiyazdan@gmail.com](mailto:salimiyazdan@gmail.com)

