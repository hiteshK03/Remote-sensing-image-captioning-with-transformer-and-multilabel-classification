# Exploring Transformer and Multi Label Classification for Remote Sensing Image Captioning

## Installation
The program requires the following dependencies:
* pytorch
* fairseq 0.9.0
* CUDA (for using GPU)

## Setup
We are using COCO Caption Evaluation library, which uses the Stanford CoreNLP 3.6.0 toolset
```
cd external/coco-caption
./get_stanford_models.sh
export PYTHONPATH=./external/coco-caption
```

## Pre-procesing
Pre-process UC Merced images and captions
```
./preprocess_captions.sh uc-merced
./preprocess_images.sh uc-merced
```

## Note
Add/Replace files to fairseq 0.9.0 from [fairseq](https://github.com/hiteshK03/Remote-sensing-image-captioning-with-transformer-and-multilabel-classification/tree/main/fairseq)  

## Training
Hyperparameters need to be tuned. This is just an example.
```
python -m fairseq_cli.train \
  --save-dir .checkpoints \
  --user-dir task \
  --task captioning \
  --arch default-captioning-arch \
  --encoder-layers 3 \
  --decoder-layers 6 \
  --features obj \
  --feature-spatial-encoding \
  --optimizer adam \
  --adam-betas "(0.9,0.999)" \
  --lr 0.0003 \
  --lr-scheduler inverse_sqrt \
  --min-lr 1e-09 \
  --warmup-init-lr 1e-8 \
  --warmup-updates 8000 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --weight-decay 0.0001 \
  --dropout 0.3 \
  --max-epoch 25 \
  --max-tokens 4096 \
  --max-source-positions 100 \
  --encoder-embed-dim 512 \
  --num-workers 2
```

## Evaluation
### Generate
To generate captions for images in test-split
```
python generate.py \
  --user-dir task \
  --features grid \
  --tokenizer moses \
  --bpe subword_nmt \
  --bpe-codes output/codes.txt \
  --beam 5 \
  --split test \
  --path .checkpoints-scst/checkpoint24.pt \
  --input output/test-ids.txt \
  --output output/test-predictions.json \
  --output_l output/test-labels-preds.csv
```
### Scoring
The following example calculates metrics for captions contained in 
`output/test-predictions.json`.

```
./score.sh \
  --reference-captions external/coco-caption/annotations/captions_val2014.json \
  --system-captions output/test-predictions.json
```

The following example calculates metrics for labels contained in 
`output/test-labels-preds.csv`.

```
python score_label.py
  --reference-captions output/label_preds.csv \
  --system-captions output/test-labels-preds.csv
```


## Model

The trained multi-task model for image captioning with multi-label classification can be downloaded from [here](https://drive.google.com/file/d/1bMoteWowfGY6pnn761X08eh4GnQbL-e6/view?usp=sharing)

## Results

Image | Caption |
--- | --- |
<img src="https://user-images.githubusercontent.com/45922320/137482373-96f92477-830f-4a8c-9180-fd847a035793.png" width="400"> | **Ground truth Caption:** This is a part of a golf course with green turfs and some bunkers and trees . <br/>**Caption w/o multi-label:** green turfs and some bunkers and withered trees in the golf course.  <br/>**Caption with multi-label:** this is a part of a golf course with green turfs and some bunkers and trees. |
<img src="https://user-images.githubusercontent.com/45922320/137482383-11ab9ee7-0158-47ec-9468-dd82478b7c5b.png" width="400"> | **Ground truth Caption:** There are two tennis courts arranged neatly and surrounded by some plants .  <br/>**Caption w/o multi-label:** four tennis courts arranged neatly with some plants surrounded.  <br/>**Caption with multi-label:** there are two tennis courts arranged neatly and surrounded by some plants. |
<img src="https://user-images.githubusercontent.com/45922320/137482371-35ec07d9-b31e-4c70-9e7e-c8a892725b0b.png" width="400"> | **Ground truth Caption:** Two straight freeways parallel forward with some cars on them .  <br/>**Caption w/o multi-label:** some cars are on the freeways.  <br/>**Caption with multi-label:** two straight freeways closed to each other with some cars on them. |
<img src="https://user-images.githubusercontent.com/45922320/137482361-8c3e53a0-ce9d-4270-b488-c343c9dff24c.png" width="400"> | **Ground truth Caption:** Two airplanes are stopped at the airport .  <br/>**Caption w/o multi-label:** an airplane is stopped at the airport.  <br/>**Caption with multi-label:** two airplanes are stopped at the airport. |
<img src="https://user-images.githubusercontent.com/45922320/137482380-f256903d-2320-4278-9474-ee214aca3ca7.png" width="400"> | **Ground truth Caption:** Many mobile homes are closed to each other with some cars parked at the roadside in the mobile home park .  <br/>**Caption w/o multi-label:** lots of mobile homes with plants surrounded in the mobile home park.  <br/>**Caption with multi-label:** many houses arranged neatly with plants surrounded in the medium residential area. |
<img src="https://user-images.githubusercontent.com/45922320/137482377-43f1fe47-ff94-44ff-ae4e-dcf034385d47.png" width="400"> | **Ground truth Caption:** An intersection with a road cross over the other roads .  <br/>**Caption w/o multi-label:** an overpass go across the roads diagonally with lawn surounded. <br/>**Caption with multi-label:** an overpass with a road go across another roads diagonally with some cars on the roads. |

## Results from other models

Image | Caption |
--- | --- |
<img src="https://user-images.githubusercontent.com/45922320/137482373-96f92477-830f-4a8c-9180-fd847a035793.png" width="400"> | **Ground truth Caption:** This is a part of a golf course with green turfs and some bunkers and trees . <br/>**Caption with angle prediction:** a part of a golf course with green turfs and some bunkers and a trail cross the turfs.  <br/>**Caption with reconstruction:** this is a part of a golf course with green turfs and some trees. |
<img src="https://user-images.githubusercontent.com/45922320/137482383-11ab9ee7-0158-47ec-9468-dd82478b7c5b.png" width="400"> | **Ground truth Caption:** There are two tennis courts arranged neatly and surrounded by some plants .  <br/>**Caption with angle prediction:** there are six tennis courts arranged neatly and surrounded by some buildings.  <br/>**Caption with reconstruction:** this is a sparse residential area with a villa surrounded by trees. |
<img src="https://user-images.githubusercontent.com/45922320/137482371-35ec07d9-b31e-4c70-9e7e-c8a892725b0b.png" width="400"> | **Ground truth Caption:** Two straight freeways parallel forward with some cars on them .  <br/>**Caption with angle prediction:** two straight freeways with some cars on them.  <br/>**Caption with reconstruction:** an overpass with a road go across another roads diagonally with some cars on the roads. |
<img src="https://user-images.githubusercontent.com/45922320/137482361-8c3e53a0-ce9d-4270-b488-c343c9dff24c.png" width="400"> | **Ground truth Caption:** Two airplanes are stopped at the airport .  <br/>**Caption with angle prediction:** it is a purple airplane stopped at the airport.  <br/>**Caption with reconstruction:** an airplane is stopped at the airport and the ground is dark. |
<img src="https://user-images.githubusercontent.com/45922320/137482380-f256903d-2320-4278-9474-ee214aca3ca7.png" width="400"> | **Ground truth Caption:** Many mobile homes are closed to each other with some cars parked at the roadside in the mobile home park .  <br/>**Caption with angle prediction:** many houses arranged in lines in the dense residential area.  <br/>**Caption with reconstruction:** lots of mobile homes with plants surrounded in the mobile home park. |
<img src="https://user-images.githubusercontent.com/45922320/137482377-43f1fe47-ff94-44ff-ae4e-dcf034385d47.png" width="400"> | **Ground truth Caption:** An intersection with a road cross over the other roads .  <br/>**Caption with angle prediction:** an overpass go across the roads with some cars on the roads. <br/>**Caption with reconstruction:** an overpass with a road go across another roads diagonally with some cars on it. |

## Reference
Codebase inspired from https://github.com/krasserm/fairseq-image-captioning  

If you find this code useful for your research, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/9855519):
```
@article{kandala2022exploring,
  title={Exploring Transformer and multi-label classification for remote sensing image captioning},
  author={Kandala, Hitesh and Saha, Sudipan and Banerjee, Biplab and Zhu, Xiao Xiang},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2022},
  publisher={IEEE}
}
```

