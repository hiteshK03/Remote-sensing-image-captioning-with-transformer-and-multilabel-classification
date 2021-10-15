# Remote Sensing Image Captioning with Transformer and Multilabel Classification
## Installation
The program requires the following dependencies:
* pytorch
* fairseq 0.9.0
* CUDA (for using GPU)

## Instructions to run code
Will be updated soon.

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
