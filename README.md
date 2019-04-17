# robust-adv-malware-detection

Welcome to the Robust Malware Detection Challenge. For more details please refer to 



This code is adapted from the code for the paper [Adversarial Deep Learning for Robust Detection of Binary Encoded Malware](https://arxiv.org/pdf/1801.02950.pdf), A. Al-Dujaili *et al.*, 2018.

The challenge has two tracks: defend and attack.

Datasets for the two tracks can be shared upon request, please email advmalwarechallenge@gmail.com and we will send you a link to the dataset.

## Installation

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


## Running:


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


## Generating Results

Results (accuracy metrics, bscn measures,  and evasion rates) will be populated under (to-be-generated) `result_files` directory. On the other hand, the trained models will be saved under `helper_files`.

The results can be compiled into LaTeX tables saved under `result_files` by runnig the function `create_tex_tables()` with the valid filepath to the result files under `utils/script_functions.py`. By default, you can do the following
```
cd utils/
python script_functions.py
```


**NOTE** For linux OS, you may run into the trouble of running `source` from within Python `os.system()`. A workaround is to replace the `os.system()` command in `run_experiments.py` with the following line:
```
system('/bin/bash -c "source activate nn_mal;python framework.py”')
```

## Citation

If you make use of this code and you'd like to cite us, please consider the following:

```
@article{al2018adversarial,
  title={Adversarial Deep Learning for Robust Detection of Binary Encoded Malware},
  author={Al-Dujaili, Abdullah and Huang, Alex and Hemberg, Erik and O'Reilly, Una-May},
  journal={arXiv preprint arXiv:1801.02950},
  year={2018}
}
```
