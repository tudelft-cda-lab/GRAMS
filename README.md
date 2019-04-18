# robust-adv-malware-detection

Welcome to the Robust Malware Detection Challenge. This challenge is organized in conjunction with the [1st Workshop on Adversarial Learning Methods for Machine Learning and Data Mining @ KDD 2019](https://sites.google.com/view/advml/).


The bulk of adversarial machine learning research has been focused on crafting attacks and defenses for image classification. In this challenge, we put adversarial machine learning in the context of robust malware detection. In the era of modern cyber warfare, cyber adversaries craft adversarial malicious code that can evade malware detectors. The problem of crafting adversarial examples in the malware classification setup is more challenging than image classifications: malware adversarial examples must not only fool the classifier, they must also ensure that their adversarial perturbations do not alter the malicious payload. The proposed task for this challenge is to defend against adversarial attacks by building robust detectors and/or attack robust malware detectors based on binary indicators of imported functions used by the malware. The challenge has two tracks:
- **Defense Track**: Build high-accuracy deep models that are robust to adversarial attacks.
- **Attack Track**: Craft adversarial malware that evades detection on adversarially trained models.

This challenge is based on the following papers.

1. [Adversarial Deep Learning for Robust Detection of Binary Encoded Malware](https://arxiv.org/pdf/1801.02950.pdf) paper's.

Datasets for the two tracks can be shared upon request, please email `advmalwarechallenge@gmail.com` and we will send you a link to the dataset. We note that the dataset contains feature vectors of the PE files not the actual binaries.

----
#### Repo Setup

##### Installation

All the required packages are specified in the yml files under `helper_files`. If you have `conda` installed, you can just `cd` to the main directory and execute the following with `osx_environment.yml` or `linux_environment.yml` on OSx or Linux, respectively. 

```
conda env create —f ./helper_files/(osx|linux)_environment.yml
```
This will create an environment called `nn_mal`.


To activate this environment, execute:
```
source activate nn_mal
```

**Note**: If you’re running the code on Mac OS with Cuda, then according to Pytorch.org
“macOS Binaries dont support CUDA, install from source if CUDA is needed”

----
##### Code Structure:

The code main's script is `framework.py`, which implements an adversarial training framework based on the paper above. 
Over multiple epochs, it fits the model using a minmax formulation (specified by the `training_method` parameter in the `parameters.ini` file) and reports the performance on a validation set. 
It also reports the performance on a test set and evaluates its robustness against an attack/inner maximizer (specified by the `evasion_method` parameter in the `parameters.ini` file).

We modified the code to support both challenge tracks. Towards the end of the `main()` function, there are 4 conditional code blocks. They can be activated based on the `[challenge]` parameters in `parameters.ini` as follows.

1. For training a model
```
[challenge]
eval = False
attack = False
defend = True
```
Note: the model can be trained from scratch or load its weights from a file as specified in the current version of the `parameters.ini`. 
```
load_model_weights = True
model_weights_path = ./helper_files/[training:natural|evasion:dfgsm_k]_demo-model.pt
```
To train it from scratch, set `load_model_weights` to `False`

2. For crafting and storing adversarial examples
```
[challenge]
eval = False
attack = True
defend = False
adv_examples_path = "PATH TO WHERE ADVERSARIAL EXAMPLES SHOULD BE STORED"
```
This setup will run the `evasion_method` specified in `parameters.ini` against the model defined by the `load_model_weights` and `model_weights_path` parameters and generate a







The rest of the repo is organized as follows.
- inner_maximizers: a module for implementing inner maximizer algorithms that satisfy the malware perturbation constraints of their binary features (e.g., 'rfgsm_k', 'bca_k' from [1]). These algorithms can be used for training and/or attacking the model based on `parameters.ini`. For instance, the current version of `parameters.ini` sets `training_method = natural` and `evasion_method = dfgsm_k`. This means if `framework.py` is made to train a model (this can be set by other parameters in `parameters.ini`), then it will train a `natural` model (no adversarial training) and on the test test, it will evaluate the model's evasion rate based on `dfgsm_k` attack. Participants can also add their own inner maximizer and modify the `training_method`/`evasion_method` in `parameters.ini` accordingly.
- nets: a module for defining the malware detector's model architecture. Currently, there is only one model architecture defined in `ff_classifier.py`. Participants may define their own model architecture in a similar format to that of `ff_classifier.py`. If you plan to change the model architecture, ensure the following:
    - The model's input has the same dimensionality of the PE feature vector.
    - The output layer should be an `nn.LogSoftmax(dim=1))` of 2 neurons.
    - Change Line 76 of `framework.py` to construct the new model
    - You may change `parameters.ini` parameters: `ff_h1`, `ff_h2`, and any model-specific parameter.
- datasets: a module for loading the dataset (malicious/benign). Participants need not to modify this.
- helper_files: contains supporting files for setting up the Python environment, PE feature mapping, and a baseline natural model `[training:natural|evasion:dfgsm_k]_demo-model.pt`. In training mode (for the defend challenge), trained models in `*.pt` format will be saved in this directory and participants of the **defend** challenge need to share these files with us for evalution.
- utils: a module that implements multiple functions for computing performance metric and texifying them.
- run_experiments.py: a script that participants may find helpful. It shows how the `parameters.ini` file can be changed programetically and running `framework.py` afterward. For instance, participants in the **attack** challenge may first train a robust model on their own, then use it to craft the transferable black-box adversarial examples. For this, `parameters.ini` must be changed from training a model to crafting adversarial examples (we will show this shortly).
- blindspot_covarge: This can be ignored. It implements an adversarial training metric from [1]

##### Instructions for the Defend Track



##### Instructions for the Attack Track

1. Configure your experiment as desired by modifying the `parameters.ini` file. Among the things you may want to to specify:
    a - dataset filepath for both malicious and benign samples
    b - gpu device if any
    c - name of the experiment 
    d - training method (inner maximizer)
    e - evasion method

**Note** In case you do not have access to the dataset, you can still run the code on a synthetic dataset with 8-dimensional binary feature vectors, whose bits are set with probability 0.2 for malicious class and 0.8 for benign class.

2. Execute `framework.py`

```
python framework.py
```
More specific instructions for each track will be provided with the datasets.



#### Tabulating Results

Results (accuracy metrics, bscn measures,  and evasion rates) will be populated under (to-be-generated) `result_files` directory. On the other hand, the trained models will be saved under `helper_files`.

The results can be compiled into LaTeX tables saved under `result_files` by runnig the function `create_tex_tables()` with the valid filepath to the result files under `utils/script_functions.py`. By default, you can do the following
```
cd utils/
python script_functions.py
```




-----
#### Citation

If you make use of this code and you'd like to cite us, please consider the following:

```
@article{al2018adversarial,
  title={Adversarial Deep Learning for Robust Detection of Binary Encoded Malware},
  author={Al-Dujaili, Abdullah and Huang, Alex and Hemberg, Erik and O'Reilly, Una-May},
  journal={arXiv preprint arXiv:1801.02950},
  year={2018}
}
```
