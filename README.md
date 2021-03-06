# GRAMS based on Robust Malware Detection Challenge as follows

added methods to`inner_maxizers.py` 

* topk
* topk+
* topkr
* rand

Methods are described in:

```
@inproceedings{verwer2020robust,
  title={The Robust Malware Detection Challenge and Greedy Random Accelerated Multi-Bit Search},
  author={Verwer, Sicco and Nadeem, Azqa and Hammerschmidt, Christian and Bliek, Laurens and Al-Dujaili, Abdullah and O'Reilly, Una-May},
  booktitle={Proceedings of the 13th ACM Workshop on Artificial Intelligence and Security},
  pages={61--70},
  year={2020}
}
``` 

and the framework is based on
```
@inproceedings{al2018adversarial,
  title={Adversarial deep learning for robust detection of binary encoded malware},
  author={Al-Dujaili, Abdullah and Huang, Alex and Hemberg, Erik and O'Reilly, Una-May},
  booktitle={2018 IEEE Security and Privacy Workshops (SPW)},
  pages={76--82},
  year={2018},
  organization={IEEE}
}
```

# Robust Malware Detection Challenge

Welcome to the Robust Malware Detection Challenge. This challenge is organized in conjunction with the [1st Workshop on Adversarial Learning Methods for Machine Learning and Data Mining @ KDD 2019](https://sites.google.com/view/advml/).


The bulk of adversarial machine learning research has been focused on crafting attacks and defenses for image classification. In this challenge, we consider adversarial machine learning in the context of robust malware detection. In the era of modern cyber warfare, cyber adversaries craft adversarial perturbations to malicious code to evade malware detectors. Crafting adversarial examples in the malware classification setup is more challenging than image classifications: malware adversarial examples must not only fool the classifier, their adversarial perturbations must not alter the malicious payload. The gist of this challenge is to defend against adversarial attacks by building robust detectors and/or attack robust malware detectors based on binary indicators of imported functions used by the malware. The challenge has two tracks:
- **Defense Track**: Build high-accuracy deep models that are robust to adversarial attacks.
- **Attack Track**: Craft adversarial **malicious** PEs that evades detection on adversarially trained models. We do not consider adversarial examples for benign PEs assuming that their authors do not intend to have them misclassified as malware.

and is based on the following paper
1. [Adversarial Deep Learning for Robust Detection of Binary Encoded Malware](https://arxiv.org/pdf/1801.02950.pdf).

Datasets for the two tracks can be shared upon request, please email `advmalwarechallenge@gmail.com` and we will send you a link to the dataset. We note that the dataset contains feature vectors of the PE files, not the actual binaries. For a formal description of the challenge, please refer to this [document](https://github.com/ALFA-group/malware_challenge/blob/master/docs/challenge.pdf).

----
##### Environment Setup

All the required packages are specified in the yml files under `helper_files`. If you have `conda` installed, you can just `cd` to the main directory and execute the following with `osx_environment.yml` or `linux_environment.yml` on OSx or Linux, respectively. 

```
conda env create —f ./helper_files/(osx|linux)_environment.yml
```
This will create an environment called `nn_mal`.


To activate this environment, execute:
```
source activate nn_mal
```

With adding the current directory to the python path:
```
export PYTHONPATH=`pwd`
```
the code is ready.

**Note**: If you’re running the code on Mac OS with Cuda, then according to Pytorch.org
“macOS Binaries dont support CUDA, install from source if CUDA is needed”

----
##### Code Structure:

The code main's script is `framework.py`, which implements an adversarial training framework based on the paper above. 
Over multiple epochs, it fits the model using a minmax formulation (specified by the `training_method` parameter in the `parameters.ini` file) and reports the performance on a validation set. 
It also reports the performance on a test set and evaluates its robustness against an attack/inner maximizer (specified by the `evasion_method` parameter in the `parameters.ini` file).

We modified the code to support both challenge tracks. Towards the end of the `main()` function, there are 4 conditional code blocks. They can be activated based on the `[challenge]` parameters in `parameters.ini` as follows.

1. For training a model (this is intended for the **defend** track, but can also be used for the **attack** track to build a subtitute model for generating transferable adversarial examples)
```
[challenge]
eval = False
attack = False
defend = True
```
, ensure that the `benign_filepath` and `malicious_filepath` point to the **defend** dataset
```
[dataset]
benign_filepath = /home/defend/benign/
malicious_filepath = /home/defend/malicious/
```
Note: the model can be trained from scratch or load its weights from a file as specified in the current version of the `parameters.ini`. 
```
load_model_weights = True
model_weights_path = ./helper_files/[training:natural|evasion:dfgsm_k]_demo-model.pt
```
To train it from scratch, set `load_model_weights` to `False`. The trained model will be stored in `./helper_files`, participants of the **defend** track need to send us the generated file.


2. For crafting and storing adversarial examples (this is intended for the **attack** track)
```
[challenge]
eval = False
attack = True
defend = False
adv_examples_path = "PATH TO WHERE ADVERSARIAL EXAMPLES SHOULD BE STORED"
```
, ensure that the `benign_filepath` and `malicious_filepath` point to the **attack** dataset; also, set the `malicious_files_list` as follows.
```
[dataset]
benign_filepath = /home/attack/benign/
malicious_filepath = /home/attack/malicious/
malicious_files_list = attack_file_list.p
```
The `malicious_files_list` parameter is set to ensure the malicious PEs are looped over in the same order across the participants (and hence the entries of the '.npy' file will correspond to the PEs in the same order).

The above setup will run the `evasion_method` specified in `parameters.ini` against the model defined by the `load_model_weights` and `model_weights_path` parameters and generate a numpy '.npy' file at `adv_examples_path`, participants of the **attack** track need to send us this file.

The last two code blocks will be used by the challenge organizers to evaluate the submissions by setting `eval = True` as shown next and handled by the script `eval_subm_script.py`. Nevertheless, participants may also use this mode to evaluate their techniques. 

3. When evaluating submissions for the **defend** track, we will use the following setup.
```
[challenge]
eval = True
attack = False
defend = True
```
, point to the holdout dataset.
```
[dataset]
benign_filepath = "PATH TO BENIGN HOLDOUT DATASET/"
malicious_filepath = "PATH TO MALICIOUS HOLDOUT DATASET/"
num_files_to_use = 3800
```
, point to the participant's submitted model
```
[general]
load_model_weights = True
model_weights_path = "PATH TO THE SUBMITTED MODEL"
```
, and we will attack the model with a set of adversaries specified by the `evasion_method` and compute the F1 score based on the performance (metrics file) against the strongest adversary.

4. When evaluating submissions for the **attack** track, we will use the following setup.
```
[challenge]
eval = True
attack = True
defend = False
adv_examples_path = "PATH TO THE SUBMITTED ADVERSARIAL EXAMPLES"
```
, point to the attack dataset.
```
[dataset]
benign_filepath = "PATH TO BENIGN ATTACK DATASET/"
malicious_filepath = "PATH TO MALICIOUS ATTACK DATASET/"
num_files_to_use = 3800
```
, and point to the secret model
```
[general]
load_model_weights = True
model_weights_path = "PATH TO THE SECRET MODEL"
```
, and we will report the evasion rate on the secret model.


The rest of the repo is organized as follows.
- inner_maximizers: a module for implementing inner maximizer algorithms that satisfy the malware perturbation constraints of their binary features (e.g., 'rfgsm_k', 'bca_k' from [1]). These algorithms can be used for training and/or attacking the model based on `parameters.ini`. For instance, the current version of `parameters.ini` sets `training_method = natural` and `evasion_method = dfgsm_k`. This means if `framework.py` is made to train a model (this can be set by other parameters in `parameters.ini`), then it will train a `natural` model (no adversarial training) and on the test test, it will evaluate the model's evasion rate based on `dfgsm_k` attack. Participants can also add their own inner maximizer and modify the `training_method`/`evasion_method` in `parameters.ini` accordingly.
- nets: a module for defining the malware detector's model architecture. Currently, there is only one model architecture defined in `ff_classifier.py`. Participants may define their own model architecture in a similar format to that of `ff_classifier.py`. If you plan to change the model architecture, ensure the following:
    - The model's input has the same dimensionality of the PE feature vector.
    - The output layer should be an `nn.LogSoftmax(dim=1))` of 2 neurons.
    - Change Line 76 of `framework.py` to construct the new model
    - You may change `parameters.ini` parameters: `ff_h1`, `ff_h2`, and any model-specific parameter.
- datasets: a module for loading the dataset (malicious/benign). Participants need not to modify this.
- run_experiments.py: a script that participants may find helpful. It shows how the `parameters.ini` file can be changed programetically and running `framework.py` afterward. For instance, participants in the **attack** challenge may first train a robust model on their own, then use it to craft the transferable black-box adversarial examples. For this, `parameters.ini` must be changed from training a model to crafting adversarial examples (we will show this shortly).
- eval_subm_script.py: the evaluation script to be used by the organizers to evaluate the submissions.
- helper_files: contains supporting files for setting up the Python environment, PE feature mapping, and a baseline natural model `[training:natural|evasion:dfgsm_k]_demo-model.pt`. In training mode (for the defend challenge), trained models in `*.pt` format will be saved in this directory and participants of the **defend** challenge need to share these files with us for evalution. The directory also has a couple of sample `.ini` files that will be used by the `eval_subm_script.py` to evaluate the submissions. You can have a look at them to understand further how `parameters.ini` can be set.
- utils: a module that implements multiple functions for computing performance metric and texifying them.
- blindspot_covarge: This can be ignored. It implements an adversarial training metric from [1]

More specific instructions for each track will be provided in the datasets' READMEs.

#### Generated Results

Results (accuracy metrics, bscn measures,  and evasion rates) will be populated under (to-be-generated) `result_files` directory every time `framework.py` is run. These json files will be used to evaluate the submissions and their names are in the form of 

```
[training:{training_method}|evasion:{evasion_method}]_{experiment_suffix}.json
```


The results can be compiled into LaTeX tables saved under `result_files` by runnig the function `create_tex_tables()` with the valid filepath to the result files under `utils/script_functions.py`. By default, you can do the following
```
cd utils/
python script_functions.py
```

As mentioned earlier, the trained models will be saved under `helper_files`.


