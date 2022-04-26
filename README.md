# Official implementation for the paper "CC-Cert: A Probabilistic Approach to Certify General Robustness of Neural Networks" 


To reproduce the results on MNIST and CIFAR-10 datasets, one should have the copies of the datasets and the weights of the networks (see below) and install the libraries:

```
pip install req.txt
```

Then, one may run the code:
```
cd estimates
python script.py --model_type plain --train_type TRAIN_TYPE --arch_type ARCHITECTURE_TYPE --dataset DATASET --base_classifier PATH_TO_CHECKPOINT --outfile SAVEFILE --split DATASET_SPLIT --batch BATCHSIZE --nsamples NUM_OF_SAMPLES --nbounds NUM_OF_BOUNDS --transform TRANSFORM_TYPE --degrees MAX_DEGREE --translation MAX_TRANSLATION_1 MAX_TRANSLATION_2 --brightness MAX_BRIGHTNESS --contrast MAX_CONTRAST --scale MAX_SCALE_1 MAX_SCALE_2 --blur_sigma MAX_BLUR_1 MAX_BLUR_2 --gamma MAX_GAMMA_FACTOR 


Parameters description

TRAIN_TYPE -- model training type, plain or with transformation-dependent augmentation; format: str; options: 'plain', 'smoothed'
ARCHITECTURE_TYPE -- architecture types with respect to the dataset, architectures are described in code/models.py; format: str; options: 'cifar_resnet110', 'mnist_43'
DATASET -- specify which dataset to certify; format: str; options: 'mnist', 'cifar10'
PATH_TO_CHECKPOINT -- path to the model checkpoint of architecture ARCHITECTURE_TYPE; format: str
SAVEFILE -- name of the text file to save the results; format: str
BATCHSIZE -- batch size; format: int
NUM_OF_SAMPLES -- number of samples used to compute a single Chernoff-Cramer bound (for a single image sample); format: int
NUM_OF_BOUNDS -- total number of Chernoff-Cramer bounds to be computed (for a single image sample); format: int
TRANSFORM_TYPE -- a transform (or composition of transforms) for a model to be certified against; format: str-str-...-str; for each 'str' (or single transform) the choice is ['blur', 'rotation', 'translation', 'gamma', 'brightness', 'scale', 'contrast']

Transformation-specific parameters below are ignored if the corresponding transfromation if not in TRANSFORM_TYPE

MAX_DEGREE -- maximum angle to rotate an image for 'rotation' transform; format: float > 0.0
MAX_TRANSLATION_1 -- maximum translation along X axis as a fraction of image X size; format: 0.0 < float < 1.0
MAX_TRANSLATION_2 -- maximum translation along Y axis as a fraction of image Y size; format: 0.0 < float < 1.0
MAX_BRIGHTNESS -- maximum brightness factor; format: float > 0.0
MAX_CONTRAST -- maximum contrast factor; format: float > 0.0
MAX_SCALE_1 -- maximum change in the scale to be appied along X axis of image; format: float >= 1.0
MAX_SCALE_2 -- maximum change in the scale to be appied along Y axis of image; format: float >= 1.0
MAX_BLUR_1 -- standard deviation of the Gaussian kernel along X axis of image; format: float > 0.0
MAX_BLUR_2 -- standard deviation of the Gaussian kernel along Y axis of image; format: float > 0.0
MAX_GAMMA_FACTOR -- maximum power of the gamma correction tansform; format: float > 0.0
```
